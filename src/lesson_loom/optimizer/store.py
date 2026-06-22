"""SQLite lineage store — a queryable record of the agent's self-improvement.

Schema (normalized so the demo can answer real questions, e.g. "rounds where a
prompt changed and held-out improved"):
  experiments(id, subject, started_at, baseline_config_id, best_config_id,
              baseline_test_northstar, best_test_northstar)
  rounds(id, experiment_id, round_num, parent_config_id, candidate_config_id,
         axis, target, rationale, promoted, train_northstar, heldout_northstar,
         train_delta, heldout_delta, goodhart_gap)
  mutations(round_id, axis, summary)
  scores(round_id, split, criterion, value)
  configs(id, json)   -- full config by content-hash id -> exact reproducibility
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from lesson_loom.core.schemas import (
    ExperimentResult,
    ExperimentRound,
    Scorecard,
    SystemConfig,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments(
  id TEXT PRIMARY KEY, subject TEXT, started_at TEXT,
  baseline_config_id TEXT, best_config_id TEXT,
  baseline_test_northstar REAL, best_test_northstar REAL);
CREATE TABLE IF NOT EXISTS rounds(
  id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, round_num INTEGER,
  parent_config_id TEXT, candidate_config_id TEXT, axis TEXT, target TEXT,
  rationale TEXT, promoted INTEGER, train_northstar REAL, heldout_northstar REAL,
  train_delta REAL, heldout_delta REAL, goodhart_gap REAL);
CREATE TABLE IF NOT EXISTS mutations(
  round_id INTEGER, axis TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS scores(
  round_id INTEGER, split TEXT, criterion TEXT, value REAL);
CREATE TABLE IF NOT EXISTS configs(id TEXT PRIMARY KEY, json TEXT);
"""


class LineageStore:
    def __init__(self, path: str | Path = "lesson_loom.sqlite"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def save_config(self, config: SystemConfig) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO configs(id, json) VALUES (?, ?)",
            (config.id, config.model_dump_json()),
        )
        self.conn.commit()

    def load_config(self, config_id: str) -> SystemConfig:
        row = self.conn.execute("SELECT json FROM configs WHERE id=?", (config_id,)).fetchone()
        if not row:
            raise KeyError(config_id)
        return SystemConfig(**json.loads(row["json"]))

    def start_experiment(self, experiment_id: str, subject: str, baseline_config_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO experiments(id, subject, started_at, baseline_config_id) "
            "VALUES (?, ?, ?, ?)",
            (experiment_id, subject, datetime.now(UTC).isoformat(), baseline_config_id),
        )
        self.conn.commit()

    def record_round(
        self,
        experiment_id: str,
        rnd: ExperimentRound,
        train_card: Scorecard,
        heldout_card: Scorecard,
    ) -> None:
        cur = self.conn.execute(
            "INSERT INTO rounds(experiment_id, round_num, parent_config_id, candidate_config_id, "
            "axis, target, rationale, promoted, train_northstar, heldout_northstar, "
            "train_delta, heldout_delta, goodhart_gap) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                experiment_id, rnd.round_num, rnd.parent_config_id, rnd.candidate_config_id,
                rnd.axis, rnd.target, rnd.rationale, int(rnd.decision.promoted),
                rnd.train_northstar, rnd.heldout_northstar,
                rnd.decision.northstar_train_delta, rnd.decision.heldout_delta,
                rnd.decision.goodhart_gap,
            ),
        )
        round_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO mutations(round_id, axis, summary) VALUES (?,?,?)",
            [(round_id, m.axis, m.summary) for m in rnd.mutations],
        )
        self.conn.executemany(
            "INSERT INTO scores(round_id, split, criterion, value) VALUES (?,?,?,?)",
            [(round_id, "train", k, v) for k, v in train_card.per_criterion_means.items()]
            + [(round_id, "heldout", k, v) for k, v in heldout_card.per_criterion_means.items()],
        )
        self.conn.commit()

    def finish_experiment(self, result: ExperimentResult) -> None:
        self.conn.execute(
            "UPDATE experiments SET best_config_id=?, baseline_test_northstar=?, "
            "best_test_northstar=? WHERE id=?",
            (
                result.best_config_id,
                result.baseline_test_northstar,
                result.best_test_northstar,
                result.experiment_id,
            ),
        )
        self.conn.commit()

    # -- queries for the CLI ---------------------------------------------------

    def latest_experiment_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT id FROM experiments ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def rounds_for(self, experiment_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM rounds WHERE experiment_id=? ORDER BY round_num", (experiment_id,)
        ).fetchall()

    def mutations_for(self, round_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT axis, summary FROM mutations WHERE round_id=?", (round_id,)
        ).fetchall()

    def close(self) -> None:
        self.conn.close()
