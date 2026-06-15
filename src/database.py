"""
src/database.py
───────────────
DuckDB in-memory analytical engine.

Responsibilities:
  • init_db()        — Spin up an in-memory DuckDB instance and register the
                       three CSV files as queryable relational tables.
  • get_db_schema()  — Reflect all loaded tables into a clean, human-readable
                       data dictionary string suitable for direct LLM injection.

Phase 2 upgrade path (Week 6):
  • init_db() will accept an optional read_only flag and enforce a
    DuckDB config that disables write operations.
  • get_db_schema() will feed directly into the ChromaDB vectorization
    pipeline (Week 5) instead of being injected raw into prompts.
"""

import os
import textwrap

import duckdb

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to this file so the module works regardless of the
# working directory the caller uses.
_SRC_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

_CSV_TABLE_MAP: dict[str, str] = {
    "users": os.path.join(_DATA_DIR, "users.csv"),
    "products": os.path.join(_DATA_DIR, "products.csv"),
    "orders": os.path.join(_DATA_DIR, "orders.csv"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> duckdb.DuckDBPyConnection:
    """
    Initialise an in-memory DuckDB connection and register all CSV files
    as named relational tables.

    Returns
    -------
    duckdb.DuckDBPyConnection
        An active, ready-to-query connection with all tables loaded.

    Raises
    ------
    FileNotFoundError
        If any expected CSV file is missing from the data/ directory.
        Run `python generate_mock_data.py` first to populate it.
    """
    for table_name, csv_path in _CSV_TABLE_MAP.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Missing data file for table '{table_name}': {csv_path}\n"
                "Run `python generate_mock_data.py` to generate the mock dataset."
            )

    conn = duckdb.connect(database=":memory:")

    for table_name, csv_path in _CSV_TABLE_MAP.items():
        # read_csv_auto infers column types automatically — clean and robust
        # for the Phase 1 3-table mock. Phase 2 will use explicit type schemas.
        conn.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')"
        )

    return conn


def get_db_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """
    Reflect all user-created tables in the connection and return a
    formatted data dictionary string ready for LLM context injection.

    The output format is deliberately plain-text (not JSON, not markdown)
    so it can be embedded into a system prompt without escaping concerns:

        Table: orders
          - order_id     (BIGINT)
          - user_id      (BIGINT)
          - product_id   (BIGINT)
          - order_date   (VARCHAR)
          - quantity     (BIGINT)
          - total_amount (DOUBLE)

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        An active DuckDB connection, typically returned by init_db().

    Returns
    -------
    str
        Multi-line plain-text schema string covering every table.
    """
    # Retrieve only base tables created by the application, excluding
    # DuckDB internal system views that appear in information_schema.
    tables_result = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_type  = 'BASE TABLE'
        ORDER BY table_name
        """
    ).fetchall()

    if not tables_result:
        return "(no tables found in the current DuckDB connection)"

    schema_blocks: list[str] = []

    for (table_name,) in tables_result:
        columns_result = conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name   = ?
            ORDER BY ordinal_position
            """,
            [table_name],
        ).fetchall()

        # Align column names for readability in the prompt
        max_col_len = max((len(col) for col, _ in columns_result), default=0)

        col_lines = [
            f"  - {col_name.ljust(max_col_len)}  ({data_type})"
            for col_name, data_type in columns_result
        ]

        block = f"Table: {table_name}\n" + "\n".join(col_lines)
        schema_blocks.append(block)

    return "\n\n".join(schema_blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run directly: python -m src.database)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initialising DuckDB connection...")
    connection = init_db()
    print("Connection established.\n")

    schema_str = get_db_schema(connection)
    print("─── Reflected Schema ───────────────────────────────────────────────")
    print(schema_str)
    print("────────────────────────────────────────────────────────────────────")

    # Sanity-check row counts
    print("\n─── Row Counts ─────────────────────────────────────────────────────")
    for table in ("users", "products", "orders"):
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    connection.close()
    print("\n[✓] database.py self-test passed.")
