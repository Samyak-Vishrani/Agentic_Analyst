import logging

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


def get_checkpointer() -> MemorySaver:
    """
    Return an initialised LangGraph checkpointer.

    Returns
    MemorySaver
        Ready for StateGraph.compile(checkpointer=...).
    """
    logger.info("[memory] Initialising MemorySaver checkpointer.")
    return MemorySaver()