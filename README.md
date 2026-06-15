# Agentic Data Analyst

A self-healing LLM-powered data analysis agent that generates SQL queries, executes them against DuckDB, and synthesizes business insights. Includes built-in error correction and optional ML forecasting.

## Features

**Core Capabilities**
- **Natural Language to SQL**: Convert user questions to DuckDB queries using LLM
- **Self-Healing**: Automatically diagnose and fix broken SQL queries (up to 3 retries)
- **Intent Classification**: Detect whether a query needs historical analysis or predictive forecasting
- **Read-Only Guardrails**: Block any destructive SQL (INSERT, DELETE, CREATE, etc.)
- **Large Dataset Handling**: Statistics-based summarization for queries > 50 rows

**Forecasting**
- **Adaptive Model Selection**: 
  - LinearRegression for < 24 months of data
  - Holt-Winters Exponential Smoothing for ≥ 24 months
- **Automatic Fallback**: HoltWinters → LinearRegression if model training fails
- **Monthly Aggregation**: Groups time series by month and predicts 3 periods ahead

**Interactive CLI**
- Real-time agent execution visibility
- SQL generation & retry tracking
- Debug mode with final state snapshots
- Schema inspection command

---

## Architecture

```
User Query (CLI)
    ↓
[Intent Classifier] → prediction_required: bool
    ↓
[SQL Generator] → generated_sql
    ↓
[SQL Executor] → data_result, data_summary
    ↓
├─→ Success: [Response Synthesizer] → AI narrative
│
├─→ Forecast Path (if prediction_required=True)
│   └─→ [Forecast Node] → model predictions
│       └─→ [Response Synthesizer] → forecast narrative
│
└─→ Error: [SQL Corrector] ⇄ [SQL Executor] (retry loop, max 3)
    └─→ [Response Synthesizer] → error message
```

**Node Ownership Map**

| Node | Keys Written |
|------|--------------|
| intent_classifier_node | `prediction_required` |
| sql_generator_node | `generated_sql` |
| sql_executor_node | `data_result`, `data_summary`, `sql_error` |
| sql_corrector_node | `generated_sql`, `retry_count`, `data_result`, `data_summary` |
| forecast_node | `forecast_result` |
| response_node | `messages` |

---

## Installation

### 1. Clone & Setup Environment

```bash
git clone <repo-url>
cd Agentic_Analyst

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### 4. Generate Mock Data

```bash
python generate_mock_data.py
```

This creates `data/orders.csv`, `data/users.csv`, and `data/products.csv` with ~500-1000 rows each.

---

## Usage

### Run the Agent

```bash
python main.py
```

### CLI Commands

| Command | Effect |
|---------|--------|
| `exit` or `quit` | Terminate session |
| `schema` | Print database schema |
| `debug on` | Show final state snapshot after each query |
| `debug off` | Hide debug output |
| User question | Execute agent pipeline |

---

## Project Structure

```
Agentic_Analyst/
├── main.py                          # CLI entry point
├── generate_mock_data.py            # Data generation (orders, users, products)
├── requirements.txt                 # Python dependencies
├── .env                             # API keys (not committed)
├── .gitignore                       # Git exclusions
├── README.md                        # This file
├── data/
│   ├── orders.csv                   # ~50k synthetic orders (2024-2026)
│   ├── users.csv                    # ~500 synthetic users
│   └── products.csv                 # ~100 synthetic products
├── src/
│   ├── __init__.py
│   ├── state.py                     # AgentState TypedDict schema
│   ├── database.py                  # DuckDB initialization & schema
│   ├── graph.py                     # LangGraph agent construction
│   ├── nodes.py                     # Agent node functions (6 nodes)
│   ├── prompts.py                   # LLM system prompts
│   ├── forecaster.py                # ML forecasting engine
│   └── forecaster_verification.ipynb # Forecast validation notebook
└── tests/                           # (Future) Unit tests
```

---

## Key Components

### `src/nodes.py` — Agent Pipeline

**Node 1: Intent Classifier**
- Parses user question
- Returns `prediction_required: bool`
- Fallback: False (assumes historical query)

**Node 2: SQL Generator**
- Injects `db_schema` into prompt
- Passes `prediction_required` flag for aliasing requirements
- Strips markdown fences from LLM output

**Node 3: SQL Executor**
- **Guardrail**: Blocks INSERT, DELETE, CREATE, DROP, etc.
- **Row Count Gate**: 
  - ≤ 50 rows → fetch all
  - > 50 rows → build statistics summary in DuckDB, fetch 5-row preview
- Returns `data_result` + `data_summary`

**Node 4: SQL Corrector**
- Diagnoses DuckDB errors
- Asks LLM to rewrite query
- Increments `retry_count` (max 3)
- Resets state fields to prevent stale data

**Node 5: Forecast Node**
- Only runs if `prediction_required=True` AND data retrieved successfully
- Calls `run_forecast()` from `src/forecaster.py`
- Returns structured `forecast_result`

**Node 6: Response Synthesizer**
- **Branch A**: Failure (retries exhausted) → polite error message
- **Branch B**: Forecast → LLM synthesizes predictions into narrative
- **Branch B-fail**: Forecast error → user-friendly error explanation
- **Branch C**: Success → LLM synthesizes data summary into narrative

### `src/forecaster.py` — ML Engine

```python
run_forecast(data_result, periods_ahead=3)
```

**Process**:
1. Parse dates → monthly periods
2. Aggregate by month (handles multiple values per day)
3. Check minimum data points (3+)
4. Select model:
   - < 24 rows → LinearRegression
   - ≥ 24 rows → HoltWinters (with fallback)
5. Predict 3 periods ahead


### `src/database.py` — DuckDB Setup

- Loads CSV files from `data/` directory
- Creates in-memory DuckDB connection
- Reflects schema for LLM consumption

**Supported Tables**:
- `orders` (order_id, user_id, product_id, order_date, total_amount, ...)
- `users` (user_id, first_name, last_name, email, ...)
- `products` (product_id, name, category, price, ...)

---

## Troubleshooting

**OPENAI_API_KEY not found**
- Create `.env` file in project root
- Format: `OPENAI_API_KEY=sk-...`
- Verify file is not in `.gitignore`

**CSV files not found**
- Run `python generate_mock_data.py` first
- Check `data/` directory exists

**DuckDB errors during SQL execution**
- Check schema with `schema` command in CLI
- Ensure column names match aliases (forecast_date, forecast_value for forecasts)
- Query has ≥ 3 months of data for forecasting

**HoltWinters forecast fails**
- Automatic fallback to LinearRegression
- May indicate insufficient data variance
- Try broader date range

---
