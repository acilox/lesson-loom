"""CLI coverage via click's runner (also exercises the report renderer)."""

from click.testing import CliRunner

from lesson_loom.cli import cli


def test_cli_generate():
    res = CliRunner().invoke(cli, ["generate", "--subject", "science"])
    assert res.exit_code == 0
    assert "Photosynthesis" in res.output


def test_cli_eval():
    res = CliRunner().invoke(cli, ["eval", "--subject", "history", "--split", "test"])
    assert res.exit_code == 0
    assert "north-star" in res.output


def test_cli_optimize_and_lineage(tmp_path):
    db = str(tmp_path / "ll.sqlite")
    report = str(tmp_path / "REPORT.md")
    res = CliRunner().invoke(
        cli, ["optimize", "--subject", "science", "--rounds", "8", "--db", db, "--report", report]
    )
    assert res.exit_code == 0, res.output
    assert "promoted" in res.output and "rejected" in res.output

    text = open(report, encoding="utf-8").read()
    assert "Gain:" in text and "Round-by-round attribution" in text and "Known limitations" in text

    res2 = CliRunner().invoke(cli, ["lineage", "--db", db])
    assert res2.exit_code == 0
    assert "experiment" in res2.output and "synthesized" in res2.output
