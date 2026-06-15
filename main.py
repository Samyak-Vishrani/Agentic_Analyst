import logging
import os
import sys
import uuid
import json

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("\n[ERROR] OPENAI_API_KEY not found in .env\n")
    sys.exit(1)

from langchain_core.messages import HumanMessage
from src.database import get_db_schema, init_db
from src.graph import build_graph


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s  [%(levelname)s]  %(name)s - %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DIVIDER = "-" * 72
THIN = "·" * 72


def _print_debug(state: dict) -> None:
    print(f"\n{THIN}")
    print("  DEBUG - Final State Snapshot")
    print(THIN)

    sql = state.get("generated_sql", "").strip()
    if sql:
        print("  SQL Executed:\n")
        for line in sql.splitlines():
            print(f"    {line}")

    print(f"\n  Retry Count : {state.get('retry_count', 0)}")
    print(f"  Prediction Flag : {state.get('prediction_required', False)}")
    print(f"  Rows in Preview : {len(state.get('data_result', []))}")

    data_summary = state.get("data_summary", "")
    if data_summary:
        print(f"\n  Stats Summary:\n")
        for line in data_summary.splitlines()[:10]:
            print(f"    {line}")

    sql_error = state.get("sql_error", "")
    if sql_error:
        print(f"\n  SQL Error: {sql_error}")

    forecast = state.get("forecast_result", {})
    if forecast and not forecast.get("error"):
        print(f"\n  Forecast: {forecast.get('model_type')} - "
              f"{len(forecast.get('predictions', []))} predictions")
        for p in forecast.get("predictions", []):
            print(f"    {p['period']}: {p['value']:,.2f}")

    preview = state.get("data_result", [])
    if preview:
        print(f"\n  Data Preview (up to 5 rows):")
        for row in preview[:5]:
            print(f"    {json.dumps(row, default=str)}")

    print(THIN)


def main() -> None:
    print(f"\n{DIVIDER}")
    print("  Agentic Data Analyst - Phase 1, Week 3")
    print("  Self-Healing SQL  |  Forecasting  |  Statistical Summaries")
    print(DIVIDER)

    print("\n  Initialising DuckDB...")
    try:
        conn = init_db()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    db_schema = get_db_schema(conn)
    print("  Database ready.\n")

    print("  Compiling LangGraph agent...")
    graph = build_graph(conn)

    # Thread ID: unique per session, enables checkpointer memory
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"  Session thread_id: {thread_id[:8]}...\n")

    print(DIVIDER)
    print("  Type a question. Commands: 'schema', 'debug on/off', 'exit'")
    print(DIVIDER + "\n")

    debug_mode = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye.\n")
            break

        if user_input.lower() == "schema":
            print(f"\n{THIN}\n{db_schema}\n{THIN}\n")
            continue

        if user_input.lower() == "debug on":
            debug_mode = True
            print("  [Debug ON]\n")
            continue

        if user_input.lower() == "debug off":
            debug_mode = False
            print("  [Debug OFF]\n")
            continue


        turn_state = {
            "messages": [HumanMessage(content=user_input)],
            "db_schema": db_schema,
            "generated_sql": "",
            "sql_error": "",
            "retry_count": 0,
            "data_result": [],
            "data_summary": "",
            "prediction_required": False,
            "forecast_result": {},
        }

        print(f"\n{THIN}")
        print("  Agent thinking...\n")

        try:
            final_state = graph.invoke(turn_state, config=config)
        except Exception as e:
            logger.error("Graph error: %s", e, exc_info=True)
            print(f"\n[ERROR] {e}\n")
            continue

        ai_messages = [
            m for m in final_state.get("messages", [])
            if hasattr(m, "type") and m.type == "ai"
        ]

        if ai_messages:
            print(f"Agent: {ai_messages[-1].content}\n")
        else:
            print("Agent: (no response generated)\n")

        if debug_mode:
            _print_debug(final_state)

        print()
        print()
        print()


if __name__ == "__main__":
    main()