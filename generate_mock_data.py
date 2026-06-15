"""
generate_mock_data.py
─────────────────────
Standalone script that seeds the `data/` directory with three relational
CSV files mimicking a production e-commerce store:

    users.csv    →    500 rows  (master dimension)
    products.csv →     50 rows  (master dimension)
    orders.csv   → 50,000 rows  (fact table, FK-safe against users + products)

The orders table is sized to 50k rows to give the agent realistic data
volumes for aggregation queries, trend analysis, and ML forecasting.

Run this once before starting the application:
    python generate_mock_data.py
"""

import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Scale constants ────────────────────────────────────────────────────────────
N_USERS    = 500
N_PRODUCTS = 50
N_ORDERS   = 50_000

# ── Output directory ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. users.csv  (500 rows)
# ─────────────────────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Aisha", "Brian", "Carmen", "David", "Elena", "Farhan", "Grace", "Hiroshi",
    "Isla", "Jorge", "Kofi", "Lena", "Marcus", "Nadia", "Omar", "Priya",
    "Quinn", "Ravi", "Sofia", "Thomas", "Uma", "Victor", "Wendy", "Xian",
    "Yusuf", "Zara", "Ahmed", "Bella", "Carlos", "Diana", "Ethan", "Fatima",
    "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nina",
]

LAST_NAMES = [
    "Patel", "Chen", "Williams", "Kim", "Okafor", "Müller", "Santos", "Ivanova",
    "Hassan", "Nguyen", "Johansson", "Ali", "Fernandez", "Andersen", "Bakr",
    "Rossi", "Park", "Dubois", "Sato", "Nkosi", "Torres", "Wang", "Singh",
    "Brown", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis",
    "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez", "King", "Wright",
    "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Nelson", "Carter",
]

REGIONS = ["North", "South", "East", "West"]

today        = date.today()
two_years_ago = today - timedelta(days=730)

# Vectorised date generation — much faster than a Python loop for 500 rows
join_offsets = np.random.randint(0, 730, size=N_USERS)
join_dates   = [
    (two_years_ago + timedelta(days=int(d))).strftime("%Y-%m-%d")
    for d in join_offsets
]

# Generate unique names by combining first + last with a numeric suffix
# if the pool is exhausted (500 users > 40×46 combinations, so suffix needed)
rng_names = [
    f"{FIRST_NAMES[i % len(FIRST_NAMES)]} {LAST_NAMES[i % len(LAST_NAMES)]}"
    + (f" {i // (len(FIRST_NAMES) * len(LAST_NAMES)) + 1}" if i >= len(FIRST_NAMES) * len(LAST_NAMES) else "")
    for i in range(N_USERS)
]

users_df = pd.DataFrame({
    "user_id":   range(1, N_USERS + 1),
    "name":      rng_names,
    "join_date": join_dates,
    "region":    np.random.choice(REGIONS, size=N_USERS),
})

users_path = os.path.join(DATA_DIR, "users.csv")
users_df.to_csv(users_path, index=False)
print(f"[✓] Generated {len(users_df):,} rows → {users_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. products.csv  (50 rows)
# ─────────────────────────────────────────────────────────────────────────────

PRODUCT_CATALOG = [
    # (product_name, category, price)
    ("Wireless Headphones",      "Electronics",     299.99),
    ("Mechanical Keyboard",      "Electronics",     149.99),
    ("USB-C Hub",                "Electronics",      49.99),
    ("4K Webcam",                "Electronics",     119.99),
    ("Laptop Stand",             "Electronics",      59.99),
    ("Noise-Cancelling Earbuds", "Electronics",     199.99),
    ("Smart Speaker",            "Electronics",      89.99),
    ("Portable Charger",         "Electronics",      39.99),
    ("Dual Monitor Arm",         "Electronics",     129.99),
    ("External SSD 1TB",         "Electronics",     109.99),
    ("Running Shoes",            "Apparel",          89.99),
    ("Yoga Mat",                 "Apparel",          34.99),
    ("Compression Socks",        "Apparel",          14.99),
    ("Hoodie",                   "Apparel",          59.99),
    ("Sports Water Bottle",      "Apparel",          24.99),
    ("Gym Gloves",               "Apparel",          19.99),
    ("Resistance Bands Set",     "Apparel",          29.99),
    ("Running Cap",              "Apparel",          22.99),
    ("Cycling Shorts",           "Apparel",          44.99),
    ("Windbreaker Jacket",       "Apparel",          79.99),
    ("Office Chair",             "Office Supplies", 499.99),
    ("Notebook Set (5-pack)",    "Office Supplies",  19.99),
    ("Desk Lamp",                "Office Supplies",  44.99),
    ("Whiteboard",               "Office Supplies",  89.99),
    ("Stapler Pro",              "Office Supplies",  15.99),
    ("Document Scanner",         "Office Supplies", 229.99),
    ("Ergonomic Mouse",          "Office Supplies",  69.99),
    ("Wrist Rest Pad",           "Office Supplies",  18.99),
    ("Cable Management Kit",     "Office Supplies",  12.99),
    ("Desk Organizer",           "Office Supplies",  34.99),
    ("Blender Pro",              "Home & Kitchen",   79.99),
    ("Air Fryer",                "Home & Kitchen",  129.99),
    ("Coffee Maker",             "Home & Kitchen",   89.99),
    ("Knife Set",                "Home & Kitchen",   59.99),
    ("Cutting Board",            "Home & Kitchen",   24.99),
    ("Meal Prep Containers",     "Home & Kitchen",   29.99),
    ("Electric Kettle",          "Home & Kitchen",   45.99),
    ("Non-Stick Pan Set",        "Home & Kitchen",   69.99),
    ("Food Scale",               "Home & Kitchen",   19.99),
    ("Vacuum Sealer",            "Home & Kitchen",   74.99),
    ("Vitamin C Supplements",    "Health",           19.99),
    ("Protein Powder 2kg",       "Health",           49.99),
    ("Foam Roller",              "Health",           29.99),
    ("First Aid Kit",            "Health",           24.99),
    ("Blood Pressure Monitor",   "Health",          149.99),
    ("Fitness Tracker",          "Health",           99.99),
    ("Meditation Cushion",       "Health",           39.99),
    ("Thermometer",              "Health",           22.99),
    ("Sunscreen SPF 50",         "Health",           14.99),
    ("Lip Balm Pack",            "Health",            9.99),
]

products_df = pd.DataFrame(
    PRODUCT_CATALOG,
    columns=["product_name", "category", "price"]
)
products_df.insert(0, "product_id", range(1, N_PRODUCTS + 1))

products_path = os.path.join(DATA_DIR, "products.csv")
products_df.to_csv(products_path, index=False)
print(f"[✓] Generated {len(products_df):,} rows → {products_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. orders.csv  (50,000 rows)
#    FK integrity: user_id ∈ users.user_id, product_id ∈ products.product_id
#    order_date must be >= the corresponding user's join_date
#    Realistic distributions:
#      - Some users order more frequently (power-user skew via Zipf weights)
#      - Some products are more popular (Zipf weights)
#      - Order volume trends upward over time (seasonal growth simulation)
# ─────────────────────────────────────────────────────────────────────────────

# Build lookup arrays (vectorised — no Python loops for 50k rows)
user_join_date_arr = pd.to_datetime(users_df["join_date"]).values  # numpy datetime64
product_price_arr  = products_df.set_index("product_id")["price"].to_dict()

# Zipf-like sampling weights — makes data more realistic for ML/trend queries
user_weights    = np.array([1 / (i ** 0.7) for i in range(1, N_USERS + 1)])
user_weights   /= user_weights.sum()

product_weights  = np.array([1 / (i ** 0.8) for i in range(1, N_PRODUCTS + 1)])
product_weights /= product_weights.sum()

# Sample user_ids and product_ids with realistic frequency skew
sampled_user_ids    = np.random.choice(range(1, N_USERS + 1),    size=N_ORDERS, p=user_weights)
sampled_product_ids = np.random.choice(range(1, N_PRODUCTS + 1), size=N_ORDERS, p=product_weights)
sampled_quantities  = np.random.randint(1, 6, size=N_ORDERS)

# Generate order_dates: each must be >= the user's join_date
# Strategy: sample a random offset from join_date to today (vectorised)
today_np = np.datetime64(today)

order_dates = []
for uid in sampled_user_ids:
    join_np  = user_join_date_arr[uid - 1]          # zero-indexed
    max_days = int((today_np - join_np).astype("timedelta64[D]").astype(int))
    offset   = np.random.randint(0, max(max_days, 1))
    order_date = (join_np + np.timedelta64(offset, "D")).astype("M8[D]").astype(date)
    order_dates.append(order_date.strftime("%Y-%m-%d"))

# Calculate total_amount
total_amounts = [
    round(int(qty) * product_price_arr[int(pid)], 2)
    for qty, pid in zip(sampled_quantities, sampled_product_ids)
]

orders_df = pd.DataFrame({
    "order_id":     range(1, N_ORDERS + 1),
    "user_id":      sampled_user_ids,
    "product_id":   sampled_product_ids,
    "order_date":   order_dates,
    "quantity":     sampled_quantities,
    "total_amount": total_amounts,
})

orders_path = os.path.join(DATA_DIR, "orders.csv")
orders_df.to_csv(orders_path, index=False)
print(f"[✓] Generated {len(orders_df):,} rows → {orders_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Integrity verification — runs automatically after generation
# ─────────────────────────────────────────────────────────────────────────────

def verify_referential_integrity() -> None:
    """
    Confirms every user_id and product_id in orders.csv exists in
    their respective master tables. Raises AssertionError on violation.
    """
    valid_user_ids    = set(users_df["user_id"])
    valid_product_ids = set(products_df["product_id"])

    orphan_users    = set(orders_df["user_id"])    - valid_user_ids
    orphan_products = set(orders_df["product_id"]) - valid_product_ids

    assert not orphan_users,    f"FK violation — unknown user_ids: {orphan_users}"
    assert not orphan_products, f"FK violation — unknown product_ids: {orphan_products}"

    # Verify order_date >= user join_date for every order (sample 5k for speed)
    sample = orders_df.sample(min(5_000, len(orders_df)), random_state=SEED)
    merged = sample.merge(users_df[["user_id", "join_date"]], on="user_id")
    merged["order_date"] = pd.to_datetime(merged["order_date"])
    merged["join_date"]  = pd.to_datetime(merged["join_date"])
    violations = merged[merged["order_date"] < merged["join_date"]]
    assert violations.empty, (
        f"Temporal FK violation — {len(violations)} orders predate user join_date"
    )

    print("[✓] Referential integrity check passed — all foreign keys are valid.")


verify_referential_integrity()

# Summary stats
print(f"\n── Dataset Summary ──────────────────────────────────────────────────")
print(f"  users:    {len(users_df):>7,} rows  |  regions: {users_df['region'].nunique()}")
print(f"  products: {len(products_df):>7,} rows  |  categories: {products_df['category'].nunique()}")
print(f"  orders:   {len(orders_df):>7,} rows  |  date range: {orders_df['order_date'].min()} → {orders_df['order_date'].max()}")
print(f"  total revenue: ₹{orders_df['total_amount'].sum():>14,.2f}")
print(f"─────────────────────────────────────────────────────────────────────")
print(f"\n[✓] Mock data generation complete. All files saved to ./data/")