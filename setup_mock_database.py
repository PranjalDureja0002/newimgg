"""
Mock Manufacturing Database Setup Script
=========================================
Creates a realistic manufacturing/supply-chain database in PostgreSQL
with ~5000+ rows of data for the Talk-to-Data POC.

Usage:
    python setup_mock_database.py

Requires:
    pip install psycopg2-binary faker

Configure the connection below or set environment variables:
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
"""

import os
import random
from datetime import datetime, timedelta

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Install psycopg2-binary: pip install psycopg2-binary")
    raise

try:
    from faker import Faker
except ImportError:
    print("Install faker: pip install faker")
    raise

fake = Faker()
Faker.seed(42)
random.seed(42)

# ---------- Configuration ----------
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "polymerdb"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "12345"),
}

# ---------- Schema DDL ----------
SCHEMA_DDL = """
-- Drop existing tables (in dependency order)
DROP VIEW  IF EXISTS monthly_production_summary CASCADE;
DROP TABLE IF EXISTS defects CASCADE;
DROP TABLE IF EXISTS quality_inspections CASCADE;
DROP TABLE IF EXISTS inventory CASCADE;
DROP TABLE IF EXISTS purchase_order_items CASCADE;
DROP TABLE IF EXISTS purchase_orders CASCADE;
DROP TABLE IF EXISTS production_orders CASCADE;
DROP TABLE IF EXISTS machines CASCADE;
DROP TABLE IF EXISTS raw_materials CASCADE;
DROP TABLE IF EXISTS suppliers CASCADE;

-- 1. Suppliers
CREATE TABLE suppliers (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    country         VARCHAR(100) NOT NULL,
    city            VARCHAR(100) NOT NULL,
    contact_email   VARCHAR(200),
    contact_phone   VARCHAR(50),
    rating          NUMERIC(3,2) CHECK (rating BETWEEN 0 AND 5),
    status          VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active','inactive','probation')),
    payment_terms   VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 2. Raw Materials
CREATE TABLE raw_materials (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    category        VARCHAR(100) NOT NULL,
    sub_category    VARCHAR(100),
    unit            VARCHAR(30) NOT NULL,
    unit_price      NUMERIC(12,2) NOT NULL,
    supplier_id     INTEGER REFERENCES suppliers(id),
    min_order_qty   INTEGER DEFAULT 1,
    lead_time_days  INTEGER DEFAULT 7,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 3. Purchase Orders
CREATE TABLE purchase_orders (
    id              SERIAL PRIMARY KEY,
    po_number       VARCHAR(30) UNIQUE NOT NULL,
    supplier_id     INTEGER REFERENCES suppliers(id),
    order_date      DATE NOT NULL,
    expected_delivery DATE,
    actual_delivery DATE,
    status          VARCHAR(30) DEFAULT 'pending'
                    CHECK (status IN ('pending','confirmed','shipped','delivered','cancelled')),
    total_amount    NUMERIC(14,2) DEFAULT 0,
    payment_status  VARCHAR(20) DEFAULT 'unpaid'
                    CHECK (payment_status IN ('unpaid','partial','paid')),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 4. Purchase Order Line Items
CREATE TABLE purchase_order_items (
    id              SERIAL PRIMARY KEY,
    po_id           INTEGER REFERENCES purchase_orders(id),
    material_id     INTEGER REFERENCES raw_materials(id),
    quantity        INTEGER NOT NULL,
    unit_price      NUMERIC(12,2) NOT NULL,
    total_price     NUMERIC(14,2) NOT NULL,
    received_qty    INTEGER DEFAULT 0,
    quality_status  VARCHAR(20) DEFAULT 'pending'
                    CHECK (quality_status IN ('pending','accepted','rejected','partial'))
);

-- 5. Machines
CREATE TABLE machines (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    machine_type    VARCHAR(100) NOT NULL,
    manufacturer    VARCHAR(200),
    model_number    VARCHAR(100),
    location        VARCHAR(100) NOT NULL,
    install_date    DATE,
    status          VARCHAR(30) DEFAULT 'operational'
                    CHECK (status IN ('operational','maintenance','idle','decommissioned')),
    last_maintenance DATE,
    next_maintenance DATE,
    efficiency_pct  NUMERIC(5,2) DEFAULT 85.00,
    operating_hours INTEGER DEFAULT 0
);

-- 6. Production Orders
CREATE TABLE production_orders (
    id              SERIAL PRIMARY KEY,
    work_order_no   VARCHAR(30) UNIQUE NOT NULL,
    product_name    VARCHAR(200) NOT NULL,
    product_code    VARCHAR(50) NOT NULL,
    machine_id      INTEGER REFERENCES machines(id),
    start_date      DATE NOT NULL,
    end_date        DATE,
    status          VARCHAR(30) DEFAULT 'planned'
                    CHECK (status IN ('planned','in_progress','completed','on_hold','cancelled')),
    quantity_planned INTEGER NOT NULL,
    quantity_produced INTEGER DEFAULT 0,
    quantity_rejected INTEGER DEFAULT 0,
    shift           VARCHAR(20) CHECK (shift IN ('morning','afternoon','night')),
    priority        VARCHAR(20) DEFAULT 'normal'
                    CHECK (priority IN ('low','normal','high','urgent')),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 7. Quality Inspections
CREATE TABLE quality_inspections (
    id                  SERIAL PRIMARY KEY,
    production_order_id INTEGER REFERENCES production_orders(id),
    inspector_name      VARCHAR(100) NOT NULL,
    inspection_date     DATE NOT NULL,
    inspection_type     VARCHAR(50) NOT NULL
                        CHECK (inspection_type IN ('in_process','final','incoming','random')),
    result              VARCHAR(20) NOT NULL
                        CHECK (result IN ('pass','fail','conditional')),
    defects_found       INTEGER DEFAULT 0,
    sample_size         INTEGER DEFAULT 0,
    notes               TEXT
);

-- 8. Inventory
CREATE TABLE inventory (
    id              SERIAL PRIMARY KEY,
    material_id     INTEGER REFERENCES raw_materials(id),
    warehouse       VARCHAR(100) NOT NULL,
    quantity_on_hand INTEGER NOT NULL DEFAULT 0,
    quantity_reserved INTEGER DEFAULT 0,
    reorder_level   INTEGER NOT NULL DEFAULT 50,
    max_stock_level INTEGER DEFAULT 1000,
    last_updated    TIMESTAMP DEFAULT NOW()
);

-- 9. Defects
CREATE TABLE defects (
    id                  SERIAL PRIMARY KEY,
    production_order_id INTEGER REFERENCES production_orders(id),
    machine_id          INTEGER REFERENCES machines(id),
    defect_type         VARCHAR(100) NOT NULL,
    severity            VARCHAR(20) NOT NULL
                        CHECK (severity IN ('minor','major','critical')),
    detected_date       DATE NOT NULL,
    detected_by         VARCHAR(100),
    root_cause          VARCHAR(200),
    resolution_status   VARCHAR(30) DEFAULT 'open'
                        CHECK (resolution_status IN ('open','investigating','resolved','closed')),
    resolution_date     DATE,
    corrective_action   TEXT
);

-- 10. Monthly Production Summary View
CREATE VIEW monthly_production_summary AS
SELECT
    DATE_TRUNC('month', po.start_date)::DATE AS month,
    COUNT(DISTINCT po.id)                     AS total_orders,
    SUM(po.quantity_planned)                  AS total_planned,
    SUM(po.quantity_produced)                 AS total_produced,
    SUM(po.quantity_rejected)                 AS total_rejected,
    ROUND(
        CASE WHEN SUM(po.quantity_planned) > 0
             THEN SUM(po.quantity_produced)::NUMERIC / SUM(po.quantity_planned) * 100
             ELSE 0 END, 2
    )                                         AS fulfillment_rate_pct,
    ROUND(
        CASE WHEN SUM(po.quantity_produced) > 0
             THEN SUM(po.quantity_rejected)::NUMERIC / SUM(po.quantity_produced) * 100
             ELSE 0 END, 2
    )                                         AS rejection_rate_pct
FROM production_orders po
GROUP BY DATE_TRUNC('month', po.start_date)
ORDER BY month;
"""

# ---------- Data Generation ----------

COUNTRIES = [
    ("India", ["Mumbai", "Delhi", "Chennai", "Pune", "Ahmedabad", "Noida"]),
    ("China", ["Shanghai", "Shenzhen", "Guangzhou", "Beijing"]),
    ("Germany", ["Munich", "Stuttgart", "Hamburg", "Berlin"]),
    ("USA", ["Detroit", "Houston", "Chicago", "Charlotte"]),
    ("Japan", ["Tokyo", "Osaka", "Nagoya"]),
    ("South Korea", ["Seoul", "Busan"]),
    ("Thailand", ["Bangkok", "Chonburi"]),
    ("Mexico", ["Monterrey", "Puebla"]),
]

MATERIAL_CATEGORIES = {
    "Polymers": {
        "sub": ["Polypropylene (PP)", "Polyethylene (PE)", "ABS Resin", "Nylon 6", "Polycarbonate", "PVC Compound"],
        "unit": "kg",
        "price_range": (80, 350),
    },
    "Additives": {
        "sub": ["UV Stabilizer", "Colorant Masterbatch", "Flame Retardant", "Impact Modifier", "Antioxidant"],
        "unit": "kg",
        "price_range": (200, 1200),
    },
    "Metal Components": {
        "sub": ["Steel Inserts", "Brass Bushings", "Aluminum Brackets", "Spring Steel Clips"],
        "unit": "pcs",
        "price_range": (5, 80),
    },
    "Packaging": {
        "sub": ["Carton Boxes", "Poly Bags", "Foam Inserts", "Stretch Wrap", "Labels"],
        "unit": "pcs",
        "price_range": (1, 25),
    },
    "Chemicals": {
        "sub": ["Mold Release Agent", "Cleaning Solvent", "Adhesive", "Surface Treatment"],
        "unit": "ltr",
        "price_range": (150, 800),
    },
}

MACHINE_TYPES = [
    ("Injection Molding", ["Engel", "Arburg", "FANUC", "Sumitomo"]),
    ("Blow Molding", ["Bekum", "Graham", "Kautex"]),
    ("Extrusion", ["KraussMaffei", "Battenfeld", "Davis-Standard"]),
    ("Assembly", ["Custom", "Automated Systems"]),
    ("CNC Machining", ["Haas", "DMG Mori", "Mazak"]),
]

PRODUCTS = [
    ("Dashboard Panel - LHD", "DP-LHD-001"),
    ("Dashboard Panel - RHD", "DP-RHD-002"),
    ("Door Trim - Front Left", "DT-FL-003"),
    ("Door Trim - Front Right", "DT-FR-004"),
    ("Door Trim - Rear Left", "DT-RL-005"),
    ("Door Trim - Rear Right", "DT-RR-006"),
    ("Bumper Assembly - Front", "BA-F-007"),
    ("Bumper Assembly - Rear", "BA-R-008"),
    ("Center Console Trim", "CC-T-009"),
    ("Pillar Trim A", "PT-A-010"),
    ("Pillar Trim B", "PT-B-011"),
    ("Pillar Trim C", "PT-C-012"),
    ("Glove Box Assembly", "GB-A-013"),
    ("Roof Liner", "RL-014"),
    ("Wheel Arch Liner", "WAL-015"),
    ("Engine Cover", "EC-016"),
    ("Fender Liner", "FEL-017"),
    ("Air Duct Assembly", "AD-018"),
    ("Cup Holder Assembly", "CH-019"),
    ("Seat Back Panel", "SBP-020"),
]

DEFECT_TYPES = [
    "Short Shot", "Flash", "Sink Marks", "Warpage", "Burn Marks",
    "Flow Lines", "Weld Lines", "Air Traps", "Delamination",
    "Color Variation", "Surface Scratches", "Dimensional Out-of-Spec",
    "Brittleness", "Contamination", "Gate Vestige",
]

LOCATIONS = ["Plant A - Line 1", "Plant A - Line 2", "Plant A - Line 3",
             "Plant B - Line 1", "Plant B - Line 2",
             "Plant C - Line 1", "Plant C - Line 2", "Plant C - Line 3"]

INSPECTORS = ["Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Verma",
              "Vikram Singh", "Deepak Mehta", "Anita Desai", "Suresh Iyer"]

WAREHOUSES = ["Main Warehouse", "Raw Material Store", "Finished Goods - A",
              "Finished Goods - B", "Quarantine Area"]

PAYMENT_TERMS = ["Net 30", "Net 45", "Net 60", "Net 90", "Advance Payment"]


def generate_suppliers(cur, count=50):
    """Generate supplier records."""
    rows = []
    for i in range(count):
        country, cities = random.choice(COUNTRIES)
        city = random.choice(cities)
        status_weights = random.choices(["active", "inactive", "probation"], weights=[80, 10, 10])[0]
        rows.append((
            fake.company(),
            country,
            city,
            fake.company_email(),
            fake.phone_number()[:20],
            round(random.uniform(2.5, 5.0), 2),
            status_weights,
            random.choice(PAYMENT_TERMS),
        ))
    execute_values(cur, """
        INSERT INTO suppliers (name, country, city, contact_email, contact_phone, rating, status, payment_terms)
        VALUES %s
    """, rows)
    print(f"  Inserted {count} suppliers")


def generate_raw_materials(cur, supplier_count=50):
    """Generate raw material records."""
    rows = []
    mat_id = 0
    for category, info in MATERIAL_CATEGORIES.items():
        for sub in info["sub"]:
            mat_id += 1
            rows.append((
                sub,
                category,
                sub.split("(")[0].strip() if "(" in sub else sub,
                info["unit"],
                round(random.uniform(*info["price_range"]), 2),
                random.randint(1, supplier_count),
                random.randint(10, 500),
                random.randint(3, 30),
            ))
    execute_values(cur, """
        INSERT INTO raw_materials (name, category, sub_category, unit, unit_price, supplier_id, min_order_qty, lead_time_days)
        VALUES %s
    """, rows)
    print(f"  Inserted {len(rows)} raw materials")
    return len(rows)


def generate_machines(cur, count=40):
    """Generate machine records."""
    rows = []
    for i in range(count):
        mtype, manufacturers = random.choice(MACHINE_TYPES)
        manufacturer = random.choice(manufacturers)
        location = random.choice(LOCATIONS)
        install_date = fake.date_between(start_date="-15y", end_date="-1y")
        last_maint = fake.date_between(start_date="-6m", end_date="today")
        next_maint = last_maint + timedelta(days=random.randint(30, 180))
        status = random.choices(
            ["operational", "maintenance", "idle", "decommissioned"],
            weights=[70, 15, 10, 5]
        )[0]
        # Older machines have lower efficiency
        years_old = (datetime.now().date() - install_date).days / 365
        base_eff = max(60, 95 - years_old * 1.5 + random.uniform(-5, 5))
        rows.append((
            f"{manufacturer} {mtype} #{i+1:03d}",
            mtype,
            manufacturer,
            f"{manufacturer[:3].upper()}-{random.randint(1000,9999)}",
            location,
            install_date,
            status,
            last_maint,
            next_maint,
            round(base_eff, 2),
            random.randint(1000, 50000),
        ))
    execute_values(cur, """
        INSERT INTO machines (name, machine_type, manufacturer, model_number, location, install_date,
                              status, last_maintenance, next_maintenance, efficiency_pct, operating_hours)
        VALUES %s
    """, rows)
    print(f"  Inserted {count} machines")


def generate_purchase_orders(cur, supplier_count=50, material_count=25, count=500):
    """Generate purchase orders with line items."""
    po_rows = []
    poi_rows = []
    base_date = datetime.now() - timedelta(days=730)  # 2 years of data

    for i in range(count):
        supplier_id = random.randint(1, supplier_count)
        order_date = base_date + timedelta(days=random.randint(0, 730))
        expected = order_date + timedelta(days=random.randint(7, 45))
        status = random.choices(
            ["pending", "confirmed", "shipped", "delivered", "cancelled"],
            weights=[10, 15, 10, 60, 5]
        )[0]
        actual = None
        if status == "delivered":
            delay = random.randint(-5, 15)
            actual = expected + timedelta(days=delay)
        payment = "paid" if status == "delivered" and random.random() > 0.2 else (
            "partial" if status in ("delivered", "shipped") and random.random() > 0.5 else "unpaid"
        )
        po_number = f"PO-{order_date.strftime('%Y%m')}-{i+1:04d}"

        # Generate 1-5 line items per PO
        num_items = random.randint(1, 5)
        po_total = 0
        for j in range(num_items):
            mat_id = random.randint(1, material_count)
            qty = random.randint(50, 5000)
            price = round(random.uniform(5, 500), 2)
            line_total = round(qty * price, 2)
            po_total += line_total
            received = qty if status == "delivered" else (
                int(qty * random.uniform(0.5, 1.0)) if status == "shipped" else 0
            )
            q_status = "accepted" if status == "delivered" else "pending"
            poi_rows.append((i + 1, mat_id, qty, price, line_total, received, q_status))

        po_rows.append((
            po_number, supplier_id, order_date.date(),
            expected.date(),
            actual.date() if actual else None,
            status, round(po_total, 2), payment,
        ))

    execute_values(cur, """
        INSERT INTO purchase_orders (po_number, supplier_id, order_date, expected_delivery,
                                     actual_delivery, status, total_amount, payment_status)
        VALUES %s
    """, po_rows)

    execute_values(cur, """
        INSERT INTO purchase_order_items (po_id, material_id, quantity, unit_price, total_price,
                                          received_qty, quality_status)
        VALUES %s
    """, poi_rows)
    print(f"  Inserted {count} purchase orders with {len(poi_rows)} line items")


def generate_production_orders(cur, machine_count=40, count=800):
    """Generate production orders over 2 years."""
    rows = []
    base_date = datetime.now() - timedelta(days=730)

    for i in range(count):
        product_name, product_code = random.choice(PRODUCTS)
        machine_id = random.randint(1, machine_count)
        start_date = base_date + timedelta(days=random.randint(0, 730))
        duration = random.randint(1, 14)
        end_date = start_date + timedelta(days=duration)
        status = random.choices(
            ["planned", "in_progress", "completed", "on_hold", "cancelled"],
            weights=[5, 10, 70, 10, 5]
        )[0]
        qty_planned = random.randint(100, 10000)
        qty_produced = int(qty_planned * random.uniform(0.85, 1.02)) if status == "completed" else (
            int(qty_planned * random.uniform(0.1, 0.6)) if status == "in_progress" else 0
        )
        qty_rejected = int(qty_produced * random.uniform(0.005, 0.08)) if qty_produced > 0 else 0
        shift = random.choice(["morning", "afternoon", "night"])
        priority = random.choices(["low", "normal", "high", "urgent"], weights=[10, 60, 20, 10])[0]
        work_order = f"WO-{start_date.strftime('%Y%m')}-{i+1:04d}"

        rows.append((
            work_order, product_name, product_code, machine_id,
            start_date.date(), end_date.date() if status != "planned" else None,
            status, qty_planned, qty_produced, qty_rejected, shift, priority,
        ))

    execute_values(cur, """
        INSERT INTO production_orders (work_order_no, product_name, product_code, machine_id,
                                       start_date, end_date, status, quantity_planned,
                                       quantity_produced, quantity_rejected, shift, priority)
        VALUES %s
    """, rows)
    print(f"  Inserted {count} production orders")


def generate_quality_inspections(cur, prod_count=800, count=1200):
    """Generate quality inspection records."""
    rows = []
    for i in range(count):
        po_id = random.randint(1, prod_count)
        inspector = random.choice(INSPECTORS)
        insp_date = fake.date_between(start_date="-2y", end_date="today")
        insp_type = random.choices(
            ["in_process", "final", "incoming", "random"],
            weights=[40, 30, 15, 15]
        )[0]
        result = random.choices(["pass", "fail", "conditional"], weights=[75, 15, 10])[0]
        defects = random.randint(0, 3) if result == "pass" else random.randint(1, 15)
        sample = random.randint(10, 200)
        notes = None
        if result != "pass":
            notes = random.choice([
                "Surface finish not meeting specifications",
                "Dimensional variance detected in batch",
                "Color inconsistency observed",
                "Material hardness below threshold",
                "Assembly fitment issues noted",
                "Minor cosmetic defects within tolerance",
                "Requires rework before shipping",
            ])
        rows.append((po_id, inspector, insp_date, insp_type, result, defects, sample, notes))

    execute_values(cur, """
        INSERT INTO quality_inspections (production_order_id, inspector_name, inspection_date,
                                         inspection_type, result, defects_found, sample_size, notes)
        VALUES %s
    """, rows)
    print(f"  Inserted {count} quality inspections")


def generate_inventory(cur, material_count=25):
    """Generate inventory records for each material in each warehouse."""
    rows = []
    for mat_id in range(1, material_count + 1):
        for wh in random.sample(WAREHOUSES, random.randint(1, 3)):
            on_hand = random.randint(0, 5000)
            reserved = int(on_hand * random.uniform(0, 0.4))
            reorder = random.randint(50, 500)
            max_stock = reorder * random.randint(5, 15)
            rows.append((mat_id, wh, on_hand, reserved, reorder, max_stock))

    execute_values(cur, """
        INSERT INTO inventory (material_id, warehouse, quantity_on_hand, quantity_reserved,
                               reorder_level, max_stock_level)
        VALUES %s
    """, rows)
    print(f"  Inserted {len(rows)} inventory records")


def generate_defects(cur, prod_count=800, machine_count=40, count=600):
    """Generate defect records with realistic correlations."""
    rows = []
    for i in range(count):
        po_id = random.randint(1, prod_count)
        machine_id = random.randint(1, machine_count)
        defect_type = random.choice(DEFECT_TYPES)
        severity = random.choices(["minor", "major", "critical"], weights=[55, 35, 10])[0]
        detected = fake.date_between(start_date="-2y", end_date="today")
        detected_by = random.choice(INSPECTORS)
        root_cause = random.choice([
            "Material batch variation", "Machine calibration drift",
            "Tooling wear", "Process parameter deviation",
            "Operator error", "Environmental conditions",
            "Raw material contamination", "Mold damage",
            "Insufficient cooling time", "Incorrect temperature settings",
        ])
        res_status = random.choices(
            ["open", "investigating", "resolved", "closed"],
            weights=[15, 15, 30, 40]
        )[0]
        res_date = None
        corrective = None
        if res_status in ("resolved", "closed"):
            res_date = detected + timedelta(days=random.randint(1, 30))
            corrective = random.choice([
                "Adjusted process parameters and validated with trial run",
                "Replaced worn tooling components",
                "Recalibrated machine sensors",
                "Updated work instructions for operators",
                "Changed material supplier batch",
                "Implemented additional in-process inspection checkpoint",
                "Modified cooling cycle time",
                "Applied preventive maintenance schedule update",
            ])

        rows.append((
            po_id, machine_id, defect_type, severity,
            detected, detected_by, root_cause, res_status, res_date, corrective,
        ))

    execute_values(cur, """
        INSERT INTO defects (production_order_id, machine_id, defect_type, severity,
                             detected_date, detected_by, root_cause, resolution_status,
                             resolution_date, corrective_action)
        VALUES %s
    """, rows)
    print(f"  Inserted {count} defect records")


def main():
    print(f"Connecting to PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    print("\n1. Creating schema...")
    cur.execute(SCHEMA_DDL)
    print("  Schema created successfully")

    print("\n2. Generating data...")
    generate_suppliers(cur, count=50)
    mat_count = generate_raw_materials(cur, supplier_count=50)
    generate_machines(cur, count=40)
    generate_purchase_orders(cur, supplier_count=50, material_count=mat_count, count=500)
    generate_production_orders(cur, machine_count=40, count=800)
    generate_quality_inspections(cur, prod_count=800, count=1200)
    generate_inventory(cur, material_count=mat_count)
    generate_defects(cur, prod_count=800, machine_count=40, count=600)

    # Print summary
    print("\n3. Data Summary:")
    tables = [
        "suppliers", "raw_materials", "machines", "purchase_orders",
        "purchase_order_items", "production_orders", "quality_inspections",
        "inventory", "defects",
    ]
    total = 0
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        total += count
        print(f"  {table:30s} {count:>6,} rows")
    print(f"  {'TOTAL':30s} {total:>6,} rows")

    print("\n4. Sample monthly production summary:")
    cur.execute("SELECT * FROM monthly_production_summary ORDER BY month DESC LIMIT 6")
    cols = [desc[0] for desc in cur.description]
    print(f"  {' | '.join(cols)}")
    print(f"  {'-' * 100}")
    for row in cur.fetchall():
        print(f"  {' | '.join(str(v) for v in row)}")

    cur.close()
    conn.close()
    print("\nDone! Database is ready for Talk-to-Data POC.")


if __name__ == "__main__":
    main()
