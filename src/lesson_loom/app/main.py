"""Thin FastAPI operator console.

Endpoints to drive and inspect the loop:
  GET  /                      dark dashboard: latest experiment leaderboard + lineage
  GET  /health
  POST /generate              {subject, content_type} -> artifact
  POST /eval                  {subject, split}        -> scorecard summary
  POST /optimize              {subject, rounds}       -> runs loop, returns summary
  GET  /api/experiments/latest
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from lesson_loom.context_packs.loader import load_pack
from lesson_loom.core.config import default_config
from lesson_loom.evals.harness import load_eval_items, run_eval
from lesson_loom.generation.graph import generate
from lesson_loom.optimizer.loop import OptimizerController
from lesson_loom.optimizer.store import LineageStore
from lesson_loom.providers import build_provider

app = FastAPI(title="lesson-loom console")
DB = os.getenv("LESSON_LOOM_DB", "lesson_loom.sqlite")


class GenerateReq(BaseModel):
    subject: str = "science"
    content_type: str = "explainer"


class EvalReq(BaseModel):
    subject: str = "science"
    split: str = "train"


class OptimizeReq(BaseModel):
    subject: str = "science"
    rounds: int = 8


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def api_generate(req: GenerateReq):
    pack = load_pack(req.subject)
    prov = build_provider("mock")
    art = generate(
        provider=prov, config=default_config(), context_pack=pack, topic=pack.topic,
        grade_level=pack.target_grade, content_type=req.content_type,
        reading_low=pack.reading_level_low, reading_high=pack.reading_level_high,
    )
    return art.model_dump()


@app.post("/eval")
def api_eval(req: EvalReq):
    pack = load_pack(req.subject)
    items = load_eval_items(req.subject, req.split)
    sc = run_eval(build_provider("mock"), default_config(), pack, items, split=req.split)
    return {"northstar": sc.northstar, "reward": sc.reward, "per_criterion": sc.per_criterion_means}


@app.post("/optimize")
def api_optimize(req: OptimizeReq):
    pack = load_pack(req.subject)
    store = LineageStore(DB)
    ctrl = OptimizerController(build_provider("mock"), pack, req.subject,
                              max_rounds=req.rounds, store=store)
    result = ctrl.run()
    store.close()
    return result.model_dump()


@app.get("/api/experiments/latest")
def api_latest():
    store = LineageStore(DB)
    exp = store.latest_experiment_id()
    if not exp:
        return JSONResponse({"experiment": None, "rounds": []})
    rounds = [dict(r) for r in store.rounds_for(exp)]
    store.close()
    return {"experiment": exp, "rounds": rounds}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    store = LineageStore(DB)
    exp = store.latest_experiment_id()
    rows_html = ""
    headline = "<p class='muted'>No experiment yet. Run <code>lesson-loom optimize</code> or POST /optimize.</p>"
    if exp:
        rounds = store.rounds_for(exp)
        for r in rounds:
            verdict = ("<span class='ok'>✅ promoted</span>" if r["promoted"]
                       else "<span class='no'>⛔ rejected</span>")
            muts = " · ".join(f"{m['axis']}: {m['summary']}" for m in store.mutations_for(r["id"]))
            rows_html += (
                f"<tr><td>{r['round_num']}</td><td>{r['axis']}</td><td>{r['target']}</td>"
                f"<td>{r['train_delta']:+.3f}</td><td>{r['heldout_delta']:+.3f}</td>"
                f"<td>{verdict}</td><td class='muted'>{r['rationale']}<br><span class='mut'>{muts}</span></td></tr>"
            )
        base = store.conn.execute(
            "SELECT baseline_test_northstar, best_test_northstar, subject FROM experiments WHERE id=?",
            (exp,),
        ).fetchone()
        if base and base["best_test_northstar"] is not None:
            gain = base["best_test_northstar"] - base["baseline_test_northstar"]
            headline = (
                f"<p><b>{base['subject']}</b> · TEST north-star "
                f"<span class='muted'>{base['baseline_test_northstar']:.3f}</span> → "
                f"<span class='ok'>{base['best_test_northstar']:.3f}</span> "
                f"<b>(+{gain:.3f})</b> <span class='muted'>frozen metric, unbiased split</span></p>"
            )
    store.close()
    return HTMLResponse(_PAGE.format(headline=headline, rows=rows_html))


_PAGE = """<!doctype html><html><head><meta charset='utf-8'><title>lesson-loom console</title>
<style>
 body{{background:#0b0d14;color:#e2e4ea;font-family:Inter,system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px}}
 h1{{font-weight:800;letter-spacing:-.02em}} h1 span{{color:#4f8cff}}
 .muted{{color:#8b90a0}} .mut{{color:#7d5cff;font-family:monospace;font-size:.8rem}}
 .ok{{color:#5bd6a0}} .no{{color:#f0a45b}}
 table{{width:100%;border-collapse:collapse;margin-top:18px}}
 th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #2a2f3f;font-size:.9rem;vertical-align:top}}
 th{{color:#8b90a0;text-transform:uppercase;font-size:.72rem;letter-spacing:.1em}}
 code{{background:#181b23;padding:2px 6px;border-radius:5px}}
</style></head><body>
 <h1>lesson<span>·</span>loom</h1>
 <p class='muted'>Self-improving educational-content engine — the agent rewrote its own prompts, refined its reward, and synthesized a check, gated by a frozen north-star.</p>
 {headline}
 <table><thead><tr><th>#</th><th>axis</th><th>target</th><th>Δtrain</th><th>Δheld-out</th><th>verdict</th><th>rationale / mutations</th></tr></thead>
 <tbody>{rows}</tbody></table>
</body></html>"""
