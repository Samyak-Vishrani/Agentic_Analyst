import json
import logging
import re
from typing import Any

import duckdb
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.forecaster import run_forecast
from src.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
    SQL_CORRECTION_PROMPT,
    SQL_GENERATION_PROMPT,
    FORECAST_SYNTHESIS_PROMPT,
)
from src.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
SMALL_RESULT_CAP = 50
PREVIEW_ROWS = 5

def _get_llm() -> ChatOpenAI:
    """ Establish connection with llm """
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _extract_previous_message(state: AgentState) -> str:
    """ Return the content of the most recent HumanMessage """

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

def _strip_markdown_fences(text: str) -> str:
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text

# Guardrails
# Only read-only mode, can't modify content
_FORBIDDEN_SQL_KEYWORDS: list[str] = [
    # DML - data mutation
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE",
    # DDL - schema mutation
    "CREATE", "DROP", "ALTER", "TRUNCATE", "RENAME",
    # DCL / admin
    "GRANT", "REVOKE", "ATTACH", "DETACH", "COPY",
    # DuckDB-specific write operations
    "EXPORT", "IMPORT", "VACUUM", "CHECKPOINT",
]

def _assert_read_only(sql: str) -> None:
    """
    Guardrail that blocks any SQL statement containing data-mutating keywords.
    Raises: PermissionError
        With a human-readable message that will be surfaced to the user via respond_node's Branch A failure path.
        The error message deliberately does NOT route through the self-healing corrector - there is no valid correction for a destructive query. 
        It should be rejected outright.
    """

    normalised = " ".join(sql.upper().split())
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalised):
            logger.warning("[guardrail] Blocked keyword '%s'.", keyword)
            raise PermissionError(
                f"Query blocked by read-only guardrail: keyword '{keyword}' "
                "is not permitted. Only SELECT queries are allowed."
            )


def _build_stats_summary(
    conn : duckdb.DuckDBPyConnection,
    sql: str,
    total_rows: int
) -> str:
    """
    Build a statistical summary of a large query result entirely inside DuckDB.

    Wraps the original SQL as a CTE and runs per-column aggregation queries directly in DuckDB

    Returns
    Plain-text summary string ready for LLM injection.
    """
    clean_sql = sql.rstrip().rstrip(";")
    # get column names and types from result schema
    try:
        meta = conn.execute(f"DESCRIBE ({clean_sql})").fetchall()
    except Exception as e:
        logger.warning("[stats] DESCRIBE failed: %s", e)
        return f"Total rows: {total_rows:,} (column metadata unavailable)"

    count_note = (
        f"Total rows: {total_rows:,} (summary statistics shown - "
        "full dataset not loaded into memory)\n"
        if total_rows > 10_000
        else f"Total rows: {total_rows:,}\n"
    )
    summary_lines = [count_note]

    for col_name, col_type, *_ in meta:
        col_upper = col_type.upper()

        # Numeric
        if any(t in col_upper for t in (
            "INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT", "HUGEINT"
        )):
            try:
                stats = conn.execute(f"""
                    SELECT
                        MIN("{col_name}") AS col_min,
                        MAX("{col_name}") AS col_max,
                        ROUND(AVG("{col_name}"), 2) AS col_mean,
                        ROUND(STDDEV("{col_name}"), 2) AS col_std
                    FROM ({clean_sql})
                """).fetchone()
                summary_lines.append(
                    f"{col_name} [numeric]: "
                    f"min={stats[0]}, max={stats[1]}, "
                    f"mean={stats[2]}, stddev={stats[3]}"
                )
            except Exception as e:
                logger.debug("[stats] Skipping numeric stats for %s: %s", col_name, e)

        # Date / timestamp
        elif any(t in col_upper for t in ("DATE", "TIMESTAMP", "TIME")):
            try:
                bounds = conn.execute(f"""
                    SELECT MIN("{col_name}"), MAX("{col_name}")
                    FROM ({clean_sql})
                """).fetchone()
                summary_lines.append(
                    f"{col_name} [date]: range {bounds[0]} → {bounds[1]}"
                )
            except Exception as e:
                logger.debug("[stats] Skipping date stats for %s: %s", col_name, e)

        # Categorical / VARCHAR
        elif any(t in col_upper for t in ("VARCHAR", "TEXT", "CHAR", "STRING")):
            try:
                top = conn.execute(f"""
                    SELECT "{col_name}", COUNT(*) AS n
                    FROM ({clean_sql})
                    GROUP BY "{col_name}"
                    ORDER BY n DESC
                    LIMIT 10
                """).fetchall()
                top_str = ", ".join([f"{r[0]}({r[1]:,})" for r in top])
                summary_lines.append(
                    f"{col_name} [categorical - top 10]: {top_str}"
                )
            except Exception as e:
                logger.debug("[stats] Skipping categorical stats for %s: %s", col_name, e)

    return "\n".join(summary_lines)



# Node 1
# Intent Classifier
def intent_classifier_node(state: AgentState) -> dict[str, Any]:
    """
    Classifies the user's intent as predictive or historical.

    Writes: prediction_required
    """

    user_ques = _extract_previous_message(state)
    llm = _get_llm()

    logger.info("[classify_intent_node] Classifying intent for: %s", user_ques)

    response = llm.invoke([
        {"role" : "system", "content" : INTENT_CLASSIFICATION_PROMPT},
        {"role" : "user" , "content" : user_ques}
    ])

    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
        prediction_required = bool(parsed.get("prediction_required", False))
    except(json.JSONDecodeError, AttributeError):
        logger.warning(
            "[classify_intent_node] Failed to parse JSON response: %r - defaulting to False",
            raw,
        )
        prediction_required = False

    logger.info("[classify_intent_node] prediction_required = %s", prediction_required)
    return {"prediction_required": prediction_required}


# Node 2
# SQL Generator
def sql_generator_node(state: AgentState) -> dict[str, Any]:
    """
    Generates a DuckDB SQL query from the user's natural language question.

    Injects the full db_schema string into the system prompt so the LLM has precise column/type information.
    Returns only the raw SQL string

    Writes: generated_sql
    """
    user_ques = _extract_previous_message(state)
    llm = _get_llm()
    db_schema = state["db_schema"]
    prediction_required = state["prediction_required"]

    logger.info("[generate_sql_node] Generating SQL for: %s", user_ques)

    system_prompt = SQL_GENERATION_PROMPT.format(
        db_schema = db_schema,
        prediction_required = prediction_required
    )

    response = llm.invoke([
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : user_ques},
    ])

    generated_sql = _strip_markdown_fences(response.content.strip())

    logger.info("[generate_sql_node] Generated SQL:\n%s", generated_sql)
    return {"generated_sql": generated_sql}



# NODE 3
# SQL Executor
def sql_executor_node(state: AgentState, conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """
    Executes the generated SQL query against the DuckDB in-memory connection.

    Writes: data_result, sql_error
    """

    sql_query = state["generated_sql"]
    logger.info("[execute_sql_node] Executing SQL:\n%s", sql_query)

    # Guardrail
    try:
        _assert_read_only(sql_query)
    except PermissionError as e:
        error_message = str(e)
        logger.warning("[execute_sql_node] Guardrail blocked query: %s", error_message)
        
        return {
            "data_result": [],
            "sql_error": error_message,
            "retry_count": MAX_RETRIES,
        }
    
    try:
        clean_sql = sql_query.rstrip().rstrip(";")

        # count rows
        count_result = conn.execute(
            f"SELECT COUNT(*) FROM ({clean_sql})"
        ).fetchone()
        total_rows = count_result[0]

        logger.info("[sql_executor_node] Row count: %d", total_rows)

        if total_rows <= SMALL_RESULT_CAP:
            # Small result - fetch everything, no summary needed
            relation = conn.execute(sql_query)
            columns = [desc[0] for desc in relation.description]
            rows = relation.fetchall()
            data_result = [dict(zip(columns, row)) for row in rows]
            data_summary = ""
            logger.info("[sql_executor_node] Small result (%d rows) - full fetch.", total_rows)

        else:
            # Large result - build stats summary inside DuckDB, keep 5-row preview
            data_summary = _build_stats_summary(conn, sql_query, total_rows)

            preview_sql = f"SELECT * FROM ({clean_sql}) LIMIT {PREVIEW_ROWS}"
            relation = conn.execute(preview_sql)
            columns = [desc[0] for desc in relation.description]
            rows = relation.fetchall()
            data_result = [dict(zip(columns, row)) for row in rows]
            logger.info(
                "[sql_executor_node] Large result (%d rows) - stats summary built, "
                "%d-row preview fetched.",
                total_rows, PREVIEW_ROWS,
            )

        return {
            "data_result": data_result,
            "data_summary": data_summary,
            "sql_error": "",
        }

    
    except Exception as e:
        error_message = str(e)
        logger.warning("[execute_sql_node] DuckDB error: %s", error_message)

        return {
            "data_result": [],
            "data_summary": "",
            "sql_error": error_message,
        }
    

# Node 4
# SQL Corrector
def sql_corrector_node(state: AgentState) -> dict[str, Any]:
    """
    Self-healing node: receives the broken SQL and raw DuckDB error, asks llm to diagnose and rewrite the query.
    The corrected SQL is written back to generated_sql
    retry_count is incremented here so the router can enforce MAX_RETRIES.

    Writes: generated_sql, retry_count, data_result
    """
    
    llm = _get_llm()
    failed_sql = state["generated_sql"]
    sql_error = state["sql_error"]
    db_schema = state["db_schema"]
    retry_count = state["retry_count"]

    new_retry_count = retry_count + 1
    logger.info(
        "[correct_sql_node] Correction attempt %d/%d for error: %s",
        new_retry_count, MAX_RETRIES, sql_error,
    )

    system_prompt = SQL_CORRECTION_PROMPT.format(
        db_schema = db_schema,
        failed_sql = failed_sql,
        sql_error = sql_error,
    )

    response = llm.invoke([
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : "Please fix the SQL query."},
    ])

    corrected_sql = _strip_markdown_fences(response.content.strip())

    logger.info("[correct_sql_node] Corrected SQL:\n%s", corrected_sql)

    return {
        "generated_sql": corrected_sql,
        "retry_count": (new_retry_count),
        "data_result": [],
        "data_summary": "",
    }


# Node 5
# Forecast Node
def forecast_node(state: AgentState) -> dict[str, Any]:
    """
    Runs the ML forecasting engine on data retrieved by sql_executor_node.
    Only reached when prediction_required=True AND sql_executor_node succeeded.

    Writes: forecast_result
    """
    data_result = state["data_result"]
    logger.info("[forecast_node] Running forecast on %d data points.", len(data_result))

    forecast_result = run_forecast(data_result, periods_ahead=3)

    if forecast_result["error"]:
        logger.warning("[forecast_node] Forecast error: %s", forecast_result["error"])
    else:
        logger.info(
            "[forecast_node] Forecast complete: %s, %d predictions.",
            forecast_result["model_type"],
            len(forecast_result["predictions"]),
        )

    return {"forecast_result": forecast_result}


# Node 6
# Respond Node
def response_node(state: AgentState) -> dict[str, Any]:
    """
    Terminal node - produces the final AI message.

    Writes: messages (add_messages reducer appends, never overwrites)
    """

    data_result = state["data_result"]
    data_summary = state["data_summary"]
    sql_error = state["sql_error"]
    prediction_required = state["prediction_required"]
    forecast_result = state.get("forecast_result", {})
    generated_sql = state["generated_sql"]
    retry_count = state["retry_count"]
    user_question = _extract_previous_message(state)

    # Branch A: Query failure
    if not data_result and not data_summary and sql_error:
        logger.warning("[response_node] Branch A - failure after %d retries.", retry_count)
        msg = (
            "I'm sorry, I encountered an internal database issue while trying to "
            "map your request to a valid query. This can happen with highly specific "
            "or complex questions.\n\n"
            f"**Technical detail:** {sql_error}\n\n"
            "Could you try rephrasing your question? Specifying exact column names "
            "or simplifying the request often helps."
        )
        return {"messages": [AIMessage(content=msg)]}

    # Branch B: Forecast path
    if prediction_required and forecast_result:

        if forecast_result.get("error"):
            logger.warning(
                "[response_node] Branch B-fail: %s", forecast_result["error"]
            )
            msg = (
                "I successfully retrieved the historical data for your forecast, "
                "but the predictive model encountered an issue:\n\n"
                f"**Detail:** {forecast_result['error']}\n\n"
                "This usually happens when there are fewer than 3 months of data "
                "or the date/value columns could not be parsed. Try asking for a "
                "broader date range or a different metric."
            )
            return {"messages": [AIMessage(content=msg)]}

        logger.info(
            "[response_node] Branch B - synthesising forecast (%s).",
            forecast_result.get("model_type"),
        )
        llm = _get_llm()

        system_prompt = FORECAST_SYNTHESIS_PROMPT.format(
            user_question = user_question,
            model_type = forecast_result.get("model_type", "unknown"),
            training_rows = forecast_result.get("training_rows", 0),
            periods_ahead = forecast_result.get("periods_ahead", 3),
            predictions = json.dumps(
                forecast_result["predictions"], indent=2, default=str
            ),
        )

        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please present the forecast findings."},
        ])
        return {"messages": [AIMessage(content=response.content.strip())]}

    # Branch C: Historical success
    logger.info("[response_node] Branch C - synthesising historical response.")
    llm = _get_llm()

    data_for_prompt = data_summary if data_summary else json.dumps(
        data_result, indent=2, default=str
    )

    system_prompt = RESPONSE_SYNTHESIS_PROMPT.format(
        user_question = user_question,
        generated_sql = generated_sql,
        data_result = data_for_prompt,
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please summarise these results."},
    ])
    return {"messages": [AIMessage(content=response.content.strip())]}
