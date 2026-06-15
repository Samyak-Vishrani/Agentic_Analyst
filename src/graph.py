import functools
import logging
from typing import Literal

import duckdb
from langgraph.graph import END, START, StateGraph

from src.memory import get_checkpointer
from src.nodes import (
    MAX_RETRIES,
    forecast_node,
    intent_classifier_node,
    response_node,
    sql_corrector_node,
    sql_executor_node,
    sql_generator_node,
)
from src.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_execution(state: AgentState) -> Literal["sql_corrector_node", "forecast_node", "response_node"]:
    sql_error   = state.get("sql_error", "")
    retry_count = state.get("retry_count", 0)
    prediction_required = state.get("prediction_required", False)

    if sql_error and retry_count < MAX_RETRIES:
        logger.info(
            "[router] SQL error detected. Routing to sql_corrector_node "
            "(attempt %d/%d).", retry_count + 1, MAX_RETRIES
        )
        return "sql_corrector_node"
    
    if sql_error and retry_count >= MAX_RETRIES:
        logger.warning(
            "[router] Max retries (%d) reached. Routing to response_node "
            "with failure state.", MAX_RETRIES
        )
        return "response_node"
    
    if prediction_required:
        logger.info("[router] Success + forecast → forecast_node.")
        return "forecast_node"
    
    logger.info("[router] SQL executed successfully. Routing to response_node.")
    return "response_node"



def build_graph(conn: duckdb.DuckDBPyConnection) -> StateGraph:
    """
    Assemble and compile the full LangGraph agent graph.
    """
    # Bind the DuckDB connection to execute_sql_node at graph compile time
    bound_execute_sql_node = functools.partial(sql_executor_node, conn=conn)
    checkpointer = get_checkpointer()

    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier_node", intent_classifier_node)
    graph.add_node("sql_generator_node", sql_generator_node)
    graph.add_node("sql_executor_node", bound_execute_sql_node)
    graph.add_node("sql_corrector_node", sql_corrector_node)
    graph.add_node("forecast_node", forecast_node)
    graph.add_node("response_node", response_node)

    graph.add_edge(START, "intent_classifier_node")
    graph.add_edge("intent_classifier_node", "sql_generator_node")
    graph.add_edge("sql_generator_node", "sql_executor_node")

    graph.add_conditional_edges(
        "sql_executor_node",
        _route_after_execution,
        {
            "sql_corrector_node": "sql_corrector_node",
            "forecast_node": "forecast_node",
            "response_node": "response_node"
        }
    )

    graph.add_edge("sql_corrector_node", "sql_executor_node")
    graph.add_edge("forecast_node", "response_node")
    graph.add_edge("response_node", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("[build_graph] Agent graph compiled successfully with MemorySaver Checkpoint.")

    return compiled