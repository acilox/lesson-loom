"""lesson-loom CLI: generate | eval | optimize | lineage | serve."""

from __future__ import annotations

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from lesson_loom.context_packs.loader import available_subjects, load_pack
from lesson_loom.core.config import default_config
from lesson_loom.core.logging import configure_logging
from lesson_loom.evals.harness import load_eval_items, run_eval
from lesson_loom.generation.graph import generate
from lesson_loom.optimizer.loop import OptimizerController
from lesson_loom.optimizer.report import render_report
from lesson_loom.optimizer.store import LineageStore
from lesson_loom.providers import build_provider

console = Console()


@click.group()
def cli() -> None:
    """A self-improving educational-content engine."""
    configure_logging()


_provider_opts = [
    click.option("--provider", default="mock", type=click.Choice(["mock", "claude"]),
                 help="mock (offline, deterministic) or claude (needs ANTHROPIC_API_KEY)."),
    click.option("--gen-model", default="claude-sonnet-4-6", help="Claude generation model."),
    click.option("--judge-model", default="claude-opus-4-8", help="Claude judge model."),
]


def _with_provider(fn):
    for opt in reversed(_provider_opts):
        fn = opt(fn)
    return fn


@cli.command()
@click.option("--subject", default="science", type=click.Choice(available_subjects()))
@click.option("--content-type", default="explainer")
@_with_provider
def generate_cmd(subject, content_type, provider, gen_model, judge_model):
    """Generate one content artifact and show its citations."""
    pack = load_pack(subject)
    prov = build_provider(provider, gen_model, judge_model)
    art = generate(
        provider=prov, config=default_config(provider_name=prov.name), context_pack=pack,
        topic=pack.topic, grade_level=pack.target_grade, content_type=content_type,
        reading_low=pack.reading_level_low, reading_high=pack.reading_level_high,
    )
    console.print(f"[bold]{pack.topic}[/bold] · {content_type} · {pack.target_grade}\n")
    console.print(escape(art.body))
    console.print(f"\n[dim]citations:[/dim] {[c.fact_id for c in art.citations]}")
    console.print(f"[dim]objectives:[/dim] {art.objectives_covered}")


cli.add_command(generate_cmd, name="generate")


@cli.command()
@click.option("--subject", default="science", type=click.Choice(available_subjects()))
@click.option("--split", default="train", type=click.Choice(["train", "heldout", "test"]))
@_with_provider
def eval_cmd(subject, split, provider, gen_model, judge_model):
    """Evaluate the baseline config on a split and print the scorecard."""
    pack = load_pack(subject)
    prov = build_provider(provider, gen_model, judge_model)
    items = load_eval_items(subject, split)
    sc = run_eval(prov, default_config(provider_name=prov.name), pack, items, split=split)
    console.print(f"[bold]{subject}/{split}[/bold]  north-star=[cyan]{sc.northstar}[/cyan]  "
                  f"reward={sc.reward}  (n={len(items)})")
    t = Table("criterion", "mean")
    for k, v in sc.per_criterion_means.items():
        t.add_row(k, f"{v:.3f}")
    console.print(t)


cli.add_command(eval_cmd, name="eval")


@cli.command()
@click.option("--subject", default="science", type=click.Choice(available_subjects()))
@click.option("--rounds", default=8, help="Max optimization rounds.")
@click.option("--strategy", default="heuristic", type=click.Choice(["heuristic", "llm"]))
@click.option("--db", default="lesson_loom.sqlite", help="SQLite lineage store path.")
@click.option("--report", "report_path", default="REPORT.md", help="Markdown report output.")
@_with_provider
def optimize(subject, rounds, strategy, db, report_path, provider, gen_model, judge_model):
    """Run the self-improvement loop and write a report + lineage store."""
    pack = load_pack(subject)
    prov = build_provider(provider, gen_model, judge_model)
    store = LineageStore(db)
    ctrl = OptimizerController(prov, pack, subject, strategy=strategy, max_rounds=rounds, store=store)
    result = ctrl.run()

    t = Table(title=f"lesson-loom · {subject} · self-improvement", show_lines=False)
    for col in ("#", "axis", "target", "Δtrain", "Δheld-out", "verdict"):
        t.add_column(col)
    for r in result.rounds:
        d = r.decision
        t.add_row(
            str(r.round_num), r.axis, r.target,
            f"{d.northstar_train_delta:+.3f}", f"{d.heldout_delta:+.3f}",
            "[green]✅ promoted[/green]" if d.promoted else "[yellow]⛔ rejected[/yellow]",
        )
    console.print(t)
    console.print(
        f"\nTEST north-star: baseline [cyan]{result.baseline_test_northstar:.3f}[/cyan] "
        f"→ best [bold green]{result.best_test_northstar:.3f}[/bold green] "
        f"([bold]+{result.test_gain:.3f}[/bold], frozen metric, unbiased split)"
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_report(result))
    console.print(f"[dim]report → {report_path} · lineage → {db}[/dim]")
    store.close()


cli.add_command(optimize, name="optimize")


@cli.command()
@click.option("--db", default="lesson_loom.sqlite")
@click.option("--experiment", default=None, help="Experiment id (defaults to the latest).")
def lineage(db, experiment):
    """Query the lineage store: show every round and what changed."""
    store = LineageStore(db)
    exp = experiment or store.latest_experiment_id()
    if not exp:
        console.print("[red]no experiments found[/red]")
        return
    console.print(f"[bold]experiment[/bold] {exp}")
    for r in store.rounds_for(exp):
        verdict = "[green]promoted[/green]" if r["promoted"] else "[yellow]rejected[/yellow]"
        console.print(
            f"  R{r['round_num']} [cyan]{r['axis']}[/cyan]/{r['target']} {verdict} "
            f"Δtrain={r['train_delta']:+.3f} Δheld={r['heldout_delta']:+.3f} "
            f"goodhart_gap={r['goodhart_gap']:+.3f}"
        )
        console.print(f"     [dim]{escape(r['rationale'])}[/dim]")
        for m in store.mutations_for(r["id"]):
            console.print(f"       · [magenta]{m['axis']}[/magenta]: {escape(m['summary'])}")
    store.close()


cli.add_command(lineage, name="lineage")


@cli.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000)
def serve(host, port):
    """Launch the FastAPI operator console."""
    import uvicorn

    uvicorn.run("lesson_loom.app.main:app", host=host, port=port, reload=False)


cli.add_command(serve, name="serve")


if __name__ == "__main__":
    cli()
