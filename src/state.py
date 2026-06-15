"""
src/state.py
────────────
LangGraph shared state definition for the Agentic Data Analyst.

This module defines AgentState — the single source of truth passed between
every node in the LangGraph execution graph. Each node reads from this state,
performs its work, and returns a partial dict to update only the keys it owns.

Design principles applied here:
  • All fields are explicitly type-annotated for IDE support and runtime
    validation via LangGraph's internal state reducers.
  • `messages` uses the built-in `add_messages` reducer so conversation
    history is *appended*, not overwritten, across multi-turn turns.
  • Every other field uses LangGraph's default reducer (last-write-wins),
    which is correct for single scalar values like sql, errors, and flags.

Phase 2 upgrade path:
  • Week 5 will add `relevant_schemas: list[str]` — the RAG-retrieved subset
    of table schemas — so the SQL generator never receives the full 40-table
    context dump.
  • Week 7 will add `cache_hit: bool` to short-circuit graph execution when
    a semantically identical query result exists in Redis.
"""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state container threaded through every node of the LangGraph agent.

    Fields
    ------
    messages : Annotated[list, add_messages]
        Full conversation history (HumanMessage + AIMessage objects).
        The `add_messages` reducer *appends* new messages rather than
        replacing the list, preserving the complete multi-turn context.

    db_schema : str
        Plain-text data dictionary produced by `src.database.get_db_schema()`.
        Injected into the SQL generation system prompt so the LLM knows
        the exact table/column structure it is querying against.
        Phase 2: this will be replaced per-query by the RAG-retrieved subset.

    generated_sql : str
        The candidate SQL string produced by the SQL generation node.
        Written by: sql_generator_node
        Read by:    sql_executor_node, sql_corrector_node (on error)
        Reset to empty string at the start of each new user turn.

    sql_error : str
        Raw database exception message captured by the executor node when
        DuckDB raises an error. Fed back verbatim into the corrector prompt
        so the LLM can diagnose and rewrite the SQL precisely.
        Empty string ("") signals a clean execution — no error.

    retry_count : int
        Monotonically increasing counter tracking how many self-correction
        attempts have been made for the current query.
        The graph routing logic uses this to enforce a hard cap of 3 retries
        before surfacing a graceful failure message to the user.
        Reset to 0 at the start of each new user turn.

    data_result : list[dict[str, Any]]
        Rows returned by a successful DuckDB query execution, serialised as
        a list of plain Python dicts (column_name → value).
        This format is JSON-serialisable and maps directly to:
          • Streamlit's st.dataframe() (Week 4)
          • Plotly chart builders (Week 4)
          • The ML forecasting engine input (Week 3)

    prediction_required : bool
        Intent flag set by the classifier node after analysing the user's
        prompt for forward-looking language (e.g., "forecast", "predict",
        "next month", "trend", "future").
        True  → graph routes through the predictive analytics node (Week 3).
        False → graph routes directly to the synthesis/response node.
    """

    messages: Annotated[list, add_messages]
    db_schema: str
    generated_sql: str
    sql_error: str
    retry_count: int
    data_result: list[dict[str, Any]]
    data_summary: str
    prediction_required: bool
    forecast_result: dict[str, Any]
