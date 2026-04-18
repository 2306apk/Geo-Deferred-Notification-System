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


def fetch_one(conn: sqlite3.Connection, query: str, params: tuple = ()) -> Dict[str, Any]:
    row = conn.execute(query, params).fetchone()
    return dict(row) if row else {}


def fetch_all(conn: sqlite3.Connection, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def build_summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    summary["overall"] = fetch_one(
        conn,
        """
        SELECT
          COUNT(*) AS total_deliveries,
          ROUND(AVG(wait_sec), 2) AS avg_wait_sec,
          ROUND(AVG(rssi), 2) AS avg_rssi_at_send,
          ROUND(AVG(quality), 3) AS avg_quality_at_send,
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
          ROUND(AVG(d.wait_sec), 2) AS avg_wait_sec,
          ROUND(AVG(d.rssi), 2) AS avg_rssi,
          ROUND(AVG(d.quality), 3) AS avg_quality,
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

    return summary


def write_markdown(summary: Dict[str, Any], out_file: Path):
    lines: List[str] = []
    lines.append("# Smart Notify Baseline Evaluation")
    lines.append("")

    overall = summary.get("overall", {})
    lines.append("## Overall")
    lines.append(f"- Total deliveries: {overall.get('total_deliveries', 0)}")
    lines.append(f"- Avg wait: {overall.get('avg_wait_sec', 'n/a')} sec")
    lines.append(f"- Avg RSSI at send: {overall.get('avg_rssi_at_send', 'n/a')} dBm")
    lines.append(f"- Avg quality at send: {overall.get('avg_quality_at_send', 'n/a')}")
    lines.append(f"- Timeout send rate: {overall.get('timeout_rate_pct', 'n/a')}%")
    lines.append(f"- Urgent send rate: {overall.get('urgent_rate_pct', 'n/a')}%")
    lines.append("")

    lines.append("## Decision Mix")
    for row in summary.get("decision_mix", []):
        lines.append(f"- {row['decision']}: {row['count']} ({row['pct']}%)")
    lines.append("")

    lines.append("## Route Breakdown")
    for row in summary.get("per_route", []):
        lines.append(
            f"- {row['route']}: deliveries={row['deliveries']}, avg_wait={row['avg_wait_sec']}s, "
            f"avg_rssi={row['avg_rssi']} dBm, avg_quality={row['avg_quality']}, "
            f"timeout_rate={row['timeout_rate_pct']}%"
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
