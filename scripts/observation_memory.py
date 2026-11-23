"""Persistent, queryable observation candidates; never confirmed control state."""
import json
import math
from pathlib import Path
import sqlite3


class ObservationMemory:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY, task TEXT NOT NULL, request TEXT NOT NULL,
            cutoff REAL NOT NULL, received REAL NOT NULL, payload TEXT NOT NULL,
            UNIQUE(task, request))""")
        self.db.execute("CREATE INDEX IF NOT EXISTS task_cutoff ON observations(task, cutoff)")
        self.db.commit()

    def append(self, task, request, cutoff, received, evidence, prediction, model_version):
        if not task or not request or not model_version:
            raise ValueError("task, request and model version required")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (cutoff, received)):
            raise ValueError("invalid times")
        if not evidence or any(type(e.get("source_last_seconds")) not in (int, float)
                               or not math.isfinite(e["source_last_seconds"])
                               or not 0 <= e["source_last_seconds"] < cutoff for e in evidence):
            raise ValueError("evidence must precede observation cutoff")
        payload = {"task": task, "request": request, "observation_cutoff_seconds": cutoff,
                   "received_at": received, "evidence": evidence, "prediction": prediction,
                   "model_version": model_version, "status": "unverified_observation_candidate",
                   "control_eligible": False}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.db:
            existing = self.db.execute("SELECT payload FROM observations WHERE task=? AND request=?", (task, request)).fetchone()
            if existing:
                if existing[0] != encoded:
                    raise ValueError("request identity reused with changed content")
                return False
            self.db.execute("INSERT INTO observations(task,request,cutoff,received,payload) VALUES (?,?,?,?,?)",
                            (task, request, cutoff, received, encoded))
        return True

    def query(self, task, *, cutoff, available_at, limit=10):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("query limit must be 1..100")
        if not math.isfinite(cutoff) or not math.isfinite(available_at):
            raise ValueError("invalid query time")
        sql = "SELECT payload FROM observations WHERE task=? AND cutoff<=?"
        parameters = [task, cutoff]
        sql += " AND received<=?"
        parameters.append(available_at)
        sql += " ORDER BY cutoff DESC, received DESC, id DESC LIMIT ?"
        return [json.loads(row[0]) for row in self.db.execute(sql, parameters+[limit])]

    def close(self):
        self.db.close()
