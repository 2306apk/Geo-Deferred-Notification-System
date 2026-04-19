#!/usr/bin/env python3
"""Generate hackathon-friendly evaluation summaries from db/smart_notify.db."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "smart_notify.db"
OUT_DIR = ROOT / "artifacts"

# Data-saving model constants (aligned with data_saving_functionality.py)
PAYLOAD_KB = 2.0
PROTOCOL_OVERHEAD = 1.15
MAX_RETRIES = 4
P_POOR = 0.82
P_GOOD = 0.02


def fetch_one(conn: sqlite3.Connection, query: str, params: tuple = ()) -> Dict[str, Any]:
    row = conn.execute(query, params).fetchone()
    return dict(row) if row else {}


def fetch_all(conn: sqlite3.Connection, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def expected_cost_kb(p_fail: float) -> float:
    """Expected transmit cost in KB using finite-retry geometric series."""
    if p_fail >= 1.0:
        return float("inf")
    attempts = (1.0 - (p_fail ** (MAX_RETRIES + 1))) / (1.0 - p_fail)
    return PAYLOAD_KB * PROTOCOL_OVERHEAD * attempts


def quality_to_p_fail(quality: float) -> float:
    """Map observed signal quality [0,1] to a failure probability."""
    q = max(0.0, min(1.0, float(quality)))
    return float(P_POOR + (P_GOOD - P_POOR) * q)


def compute_strategy_comparison(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Compare SMART strategy against NAIVE immediate send.

    Preferred baseline:
      - naive cost from notification creation quality (if available).
      - smart cost from delivery quality.
    Fallback baseline:
      - assume naive sends happen in poor coverage (P_POOR).
    """
    has_notifications_table = bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='notifications'"
        ).fetchone()
    )

    if has_notifications_table:
        rows = fetch_all(
            conn,
            """
            SELECT
              d.run_id,
              d.notif_id,
              d.wait_sec,
              d.quality AS delivered_quality,
              d.decision,
              n.created_quality,
              n.priority
            FROM deliveries d
            LEFT JOIN notifications n
              ON n.run_id = d.run_id AND n.notif_id = d.notif_id
            """,
        )
    else:
        rows = fetch_all(
            conn,
            """
            SELECT
              d.run_id,
              d.notif_id,
              d.wait_sec,
              d.quality AS delivered_quality,
              d.decision,
              NULL AS created_quality,
              NULL AS priority
            FROM deliveries d
            """,
        )

    if not rows:
        return {
            "sample_size": 0,
            "method": "none",
            "smart_total_kb": 0.0,
            "naive_total_kb": 0.0,
            "saved_kb": 0.0,
            "saved_pct": 0.0,
            "avg_quality_uplift": 0.0,
            "deferred_count": 0,
        }

    rows_with_creation = [r for r in rows if r.get("created_quality") is not None]
    eval_rows = rows_with_creation if rows_with_creation else rows

    def _aggregate(rows_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        smart_total = 0.0
        naive_total = 0.0
        quality_uplifts: List[float] = []
        deferred_count = 0

        for r in rows_subset:
            delivered_quality = float(r.get("delivered_quality") or 0.0)
            p_smart = quality_to_p_fail(delivered_quality)
            smart_total += expected_cost_kb(p_smart)

            if (r.get("wait_sec") or 0.0) > 0:
                deferred_count += 1

            created_quality = r.get("created_quality")
            if created_quality is not None:
                c_quality = float(created_quality)
                naive_total += expected_cost_kb(quality_to_p_fail(c_quality))
                quality_uplifts.append(delivered_quality - c_quality)
            else:
                naive_total += expected_cost_kb(P_POOR)

        saved = naive_total - smart_total
        saved_pct = (100.0 * saved / naive_total) if naive_total > 0 else 0.0

        return {
            "sample_size": int(len(rows_subset)),
            "smart_total_kb": round(float(smart_total), 2),
            "naive_total_kb": round(float(naive_total), 2),
            "saved_kb": round(float(saved), 2),
            "saved_pct": round(float(saved_pct), 2),
            "avg_quality_uplift": round(float(sum(quality_uplifts) / len(quality_uplifts)), 4) if quality_uplifts else 0.0,
            "deferred_count": int(deferred_count),
        }

    all_stats = _aggregate(eval_rows)

    non_urgent_rows = [
        r for r in eval_rows
        if (r.get("priority") is not None and int(r.get("priority")) > 2)
    ]
    non_urgent_stats = _aggregate(non_urgent_rows) if non_urgent_rows else {
        "sample_size": 0,
        "smart_total_kb": 0.0,
        "naive_total_kb": 0.0,
        "saved_kb": 0.0,
        "saved_pct": 0.0,
        "avg_quality_uplift": 0.0,
        "deferred_count": 0,
    }

    method = "creation_quality" if rows_with_creation else "assume_poor_baseline"

    return {
        "sample_size": int(len(eval_rows)),
        "method": method,
        "samples_with_creation_quality": int(len(rows_with_creation)),
        "smart_total_kb": all_stats["smart_total_kb"],
        "naive_total_kb": all_stats["naive_total_kb"],
        "saved_kb": all_stats["saved_kb"],
        "saved_pct": all_stats["saved_pct"],
        "avg_quality_uplift": all_stats["avg_quality_uplift"],
        "deferred_count": all_stats["deferred_count"],
        "non_urgent": non_urgent_stats,
    }


def compute_strategy_by_route(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Compute Smart-vs-Naive expected data cost per route."""
    rows = fetch_all(
        conn,
        """
        SELECT
          r.route,
          d.quality AS delivered_quality,
          d.wait_sec,
          n.created_quality,
          n.priority
        FROM deliveries d
        JOIN simulation_runs r ON r.id = d.run_id
        LEFT JOIN notifications n
          ON n.run_id = d.run_id AND n.notif_id = d.notif_id
        """,
    )

    if not rows:
        return []

    by_route: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_route.setdefault(str(r.get("route") or "unknown"), []).append(r)

    out: List[Dict[str, Any]] = []
    for route, route_rows in by_route.items():
        smart_total = 0.0
        naive_total = 0.0
        deferred_count = 0

        for rr in route_rows:
            delivered_quality = float(rr.get("delivered_quality") or 0.0)
            smart_total += expected_cost_kb(quality_to_p_fail(delivered_quality))

            if (rr.get("wait_sec") or 0.0) > 0:
                deferred_count += 1

            created_quality = rr.get("created_quality")
            if created_quality is not None:
                naive_total += expected_cost_kb(quality_to_p_fail(float(created_quality)))
            else:
                naive_total += expected_cost_kb(P_POOR)

        saved = naive_total - smart_total
        saved_pct = (100.0 * saved / naive_total) if naive_total > 0 else 0.0
        out.append(
            {
                "route": route,
                "sample_size": int(len(route_rows)),
                "smart_total_kb": round(float(smart_total), 2),
                "naive_total_kb": round(float(naive_total), 2),
                "saved_kb": round(float(saved), 2),
                "saved_pct": round(float(saved_pct), 2),
                "deferred_count": int(deferred_count),
            }
        )

    out.sort(key=lambda x: x["sample_size"], reverse=True)
    return out


def build_summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    summary["overall"] = fetch_one(
        conn,
        """
        SELECT
          COUNT(*) AS total_deliveries,
          ROUND(100.0 * SUM(CASE WHEN decision='SEND_TIMEOUT' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2)
            AS timeout_rate_pct,
          ROUND(100.0 * SUM(CASE WHEN decision='SEND_URGENT' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2)
            AS urgent_rate_pct
        FROM deliveries
        """,
    )

    summary["decision_mix"] = fetch_all(
        conn,
        """
        SELECT
          decision,
          COUNT(*) AS count,
          ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM deliveries
        GROUP BY decision
        ORDER BY count DESC
        """,
    )

    summary["per_route"] = fetch_all(
        conn,
        """
        SELECT
          r.route,
          COUNT(*) AS deliveries,
          ROUND(100.0 * SUM(CASE WHEN d.decision='SEND_TIMEOUT' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2)
            AS timeout_rate_pct
        FROM deliveries d
        JOIN simulation_runs r ON r.id = d.run_id
        GROUP BY r.route
        ORDER BY deliveries DESC
        """,
    )

    summary["recent_runs"] = fetch_all(
        conn,
        """
        SELECT id, route, started_at, ended_at, metrics
        FROM simulation_runs
        ORDER BY started_at DESC
        LIMIT 10
        """,
    )

    summary["strategy_comparison"] = compute_strategy_comparison(conn)
    summary["strategy_by_route"] = compute_strategy_by_route(conn)

    return summary


def write_markdown(summary: Dict[str, Any], out_file: Path):
    lines: List[str] = []
    lines.append("# Smart Notify Baseline Evaluation")
    lines.append("")

    overall = summary.get("overall", {})
    lines.append("## Overall")
    lines.append(f"- Total deliveries: {overall.get('total_deliveries', 0)}")
    lines.append(f"- Timeout send rate: {overall.get('timeout_rate_pct', 'n/a')}%")
    lines.append(f"- Urgent send rate: {overall.get('urgent_rate_pct', 'n/a')}%")
    lines.append("")

    comp = summary.get("strategy_comparison", {})
    lines.append("## Smart vs Naive Comparison")
    lines.append(f"- Method: {comp.get('method', 'n/a')}")
    lines.append(f"- Sample size: {comp.get('sample_size', 0)} delivered notifications")
    lines.append(f"- Smart total expected data: {comp.get('smart_total_kb', 0)} KB")
    lines.append(f"- Naive total expected data: {comp.get('naive_total_kb', 0)} KB")
    lines.append(f"- Expected data saved: {comp.get('saved_kb', 0)} KB ({comp.get('saved_pct', 0)}%)")
    lines.append(f"- Avg signal-quality uplift at send: {comp.get('avg_quality_uplift', 0)}")
    lines.append(f"- Deferred notifications: {comp.get('deferred_count', 0)}")
    non_urgent = comp.get("non_urgent", {})
    if non_urgent:
        lines.append("- Non-urgent only:")
        lines.append(
            f"  sample={non_urgent.get('sample_size', 0)}, "
            f"saved={non_urgent.get('saved_kb', 0)} KB ({non_urgent.get('saved_pct', 0)}%), "
            f"deferred={non_urgent.get('deferred_count', 0)}"
        )
    lines.append("")

    lines.append("## Decision Mix")
    for row in summary.get("decision_mix", []):
        lines.append(f"- {row['decision']}: {row['count']} ({row['pct']}%)")
    lines.append("")

    lines.append("## Route Breakdown")
    by_route_saved = {
        row["route"]: row
        for row in summary.get("strategy_by_route", [])
    }
    for row in summary.get("per_route", []):
        saved_row = by_route_saved.get(row["route"], {})
        lines.append(
            f"- {row['route']}: deliveries={row['deliveries']}, "
            f"timeout_rate={row['timeout_rate_pct']}%, "
            f"saved={saved_row.get('saved_kb', 0)} KB ({saved_row.get('saved_pct', 0)}%)"
        )

    out_file.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_PATH), help="Path to SQLite DB")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    args = parser.parse_args()

    db = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        summary = build_summary(conn)
    finally:
        conn.close()

    json_file = out_dir / "baseline_summary.json"
    md_file = out_dir / "baseline_summary.md"

    json_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, md_file)

    print(f"[Eval] Wrote {json_file}")
    print(f"[Eval] Wrote {md_file}")


if __name__ == "__main__":
    main()
