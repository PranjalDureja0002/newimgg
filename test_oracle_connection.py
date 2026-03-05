"""
Oracle Connectivity Test Script
================================
Tests connection to Oracle DB and validates the VW_SPEND_REPORT_VIEW.

Usage:
    pip install oracledb
    python test_oracle_connection.py
"""

import sys

try:
    import oracledb
except ImportError:
    print("ERROR: oracledb not installed. Run: pip install oracledb")
    sys.exit(1)


# ===== CONFIGURATION — UPDATE THESE =====
HOST = "DEBOEORACLE03.SMP.xxx.com"   # <-- update
PORT = 1523
SERVICE_NAME = "abc"
USERNAME = "EIS_READER"              # <-- update
PASSWORD = "your_password_here"      # <-- update
VIEW_NAME = "VW_SPEND_REPORT_VIEW"
# =========================================


def main():
    print("=" * 60)
    print("Oracle Connectivity Test")
    print("=" * 60)

    # 1. Connect
    print(f"\n[1/6] Connecting to {HOST}:{PORT}/{SERVICE_NAME} ...")
    dsn = oracledb.makedsn(HOST, PORT, service_name=SERVICE_NAME)
    try:
        conn = oracledb.connect(user=USERNAME, password=PASSWORD, dsn=dsn)
        conn.call_timeout = 60000  # 60s timeout
        print(f"  OK — Connected. DB version: {conn.version}")
    except Exception as e:
        print(f"  FAILED — {e}")
        sys.exit(1)

    cur = conn.cursor()

    # 2. Basic sanity
    print("\n[2/6] Basic sanity check (SELECT 1 FROM DUAL) ...")
    try:
        cur.execute("SELECT 1 FROM DUAL")
        row = cur.fetchone()
        print(f"  OK — Result: {row[0]}")
    except Exception as e:
        print(f"  FAILED — {e}")

    # 3. Check if the view exists
    print(f"\n[3/6] Checking if {VIEW_NAME} exists ...")
    try:
        cur.execute(
            "SELECT COUNT(*) FROM user_tab_columns WHERE table_name = :1",
            [VIEW_NAME],
        )
        col_count = cur.fetchone()[0]
        if col_count > 0:
            print(f"  OK — View exists with {col_count} columns")
        else:
            # Try ALL_TAB_COLUMNS in case it's in another schema
            cur.execute(
                "SELECT owner, COUNT(*) FROM all_tab_columns WHERE table_name = :1 GROUP BY owner",
                [VIEW_NAME],
            )
            rows = cur.fetchall()
            if rows:
                for owner, cnt in rows:
                    print(f"  FOUND in schema {owner} with {cnt} columns (not in current user's schema)")
            else:
                print(f"  NOT FOUND — {VIEW_NAME} does not exist in any accessible schema")
    except Exception as e:
        print(f"  FAILED — {e}")

    # 4. List all columns
    print(f"\n[4/6] Listing columns of {VIEW_NAME} ...")
    try:
        cur.execute(
            """SELECT column_name, data_type, data_length, nullable
               FROM user_tab_columns
               WHERE table_name = :1
               ORDER BY column_id""",
            [VIEW_NAME],
        )
        columns = cur.fetchall()
        if columns:
            print(f"  {'#':<4} {'COLUMN_NAME':<35} {'TYPE':<15} {'LEN':<8} {'NULL?'}")
            print(f"  {'—'*4} {'—'*35} {'—'*15} {'—'*8} {'—'*5}")
            for i, (col, dtype, dlen, nullable) in enumerate(columns, 1):
                print(f"  {i:<4} {col:<35} {dtype:<15} {dlen:<8} {nullable}")
            print(f"\n  Total: {len(columns)} columns")
        else:
            print("  No columns found (view may be in a different schema)")
    except Exception as e:
        print(f"  FAILED — {e}")

    # 5. Sample rows
    print(f"\n[5/6] Fetching 5 sample rows ...")
    try:
        cur.execute(f"SELECT * FROM {VIEW_NAME} FETCH FIRST 5 ROWS ONLY")
        col_names = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        print(f"  Fetched {len(rows)} rows, {len(col_names)} columns\n")
        for i, row in enumerate(rows):
            print(f"  --- Row {i + 1} ---")
            for name, val in zip(col_names, row):
                display_val = str(val)[:80] if val is not None else "NULL"
                print(f"    {name:<35} {display_val}")
            print()
    except Exception as e:
        print(f"  FAILED — {e}")

    # 6. Row count (with timeout protection)
    print(f"[6/6] Estimating row count (may take a moment on 200M rows) ...")
    try:
        # Use a fast approximation first
        cur.execute(f"""
            SELECT num_rows FROM all_tables
            WHERE table_name = :1 AND ROWNUM = 1
        """, [VIEW_NAME])
        est_row = cur.fetchone()
        if est_row and est_row[0]:
            print(f"  Estimated rows (from statistics): {est_row[0]:,}")
        else:
            # Fall back to COUNT with a short timeout
            print("  No statistics available — running COUNT(*) with 30s timeout ...")
            old_timeout = conn.call_timeout
            conn.call_timeout = 30000
            try:
                cur.execute(f"SELECT COUNT(*) FROM {VIEW_NAME}")
                count = cur.fetchone()[0]
                print(f"  Exact row count: {count:,}")
            except Exception as e:
                print(f"  COUNT timed out (expected for 200M rows) — {e}")
            finally:
                conn.call_timeout = old_timeout
    except Exception as e:
        print(f"  FAILED — {e}")

    # Cleanup
    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print("DONE — Connection and view validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
