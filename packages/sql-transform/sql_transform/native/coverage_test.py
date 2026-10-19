from sql_transform.native import coverage


def test_the_coverage_table_is_current():
    assert (
        coverage.block(coverage.DOC.read_text(encoding="utf-8")) == coverage.table()
    ), "regenerate it: uv run python -m sql_transform.native.coverage --write"


def test_every_named_class_has_a_row():
    # sklearn transformers, and Pipeline: a composition the catalog serves.
    listed = {n for n, _, _ in coverage.rows()}
    assert set(coverage.COMPOSITIONS) <= listed
    assert set(coverage.OUT_OF_SCOPE) <= listed


def test_a_bound_reads_by_its_shape():
    from sql_transform.native import Entry, ErrorScale

    def t(est, x, types):
        return x

    assert coverage._exactness(Entry(t, 0)) == "bit-exact"
    assert (
        coverage._exactness(Entry(t, 4))
        == "within 4 ulps, served only with `allow_bound=True`"
    )
    assert (
        coverage._exactness(Entry(t, 3, lambda est: 0))
        == "bit-exact; within 3 ulps for some configurations,"
        " served only with `allow_bound=True`"
    )
    scale = ErrorScale(lambda est: 4, lambda est, x: x, lambda est: 0)
    assert (
        coverage._exactness(Entry(t, 0, scale=scale))
        == "within K·eps·S + τ (S its error scale), served only with `allow_bound=True`"
    )
    assert (
        coverage._exactness(Entry(t, 0, scale=scale, scaled_of=lambda est: False))
        == "bit-exact; within K·eps·S + τ (S its error scale) for some"
        " configurations, served only with `allow_bound=True`"
    )


def test_nonzero_ulp_bounds_counts_the_classes_with_a_bound():
    rows = {n: note for n, s, note in coverage.rows() if s == "native"}
    assert coverage.nonzero_ulp_bounds() == sum(
        1 for note in rows.values() if note != "bit-exact"
    )


def test_the_approved_bounds_manifest_is_current():
    assert coverage.BOUNDS_DOC.read_text(encoding="utf-8") == coverage.bounds_table(), (
        "regenerate it: python -m sql_transform.native.coverage --write"
    )


def test_the_bound_guard_detects_t54_s_scale_change_without_a_k_increase():
    from scripts.check_bounds import parse, relaxations

    before = parse(coverage.BOUNDS_DOC.read_text(encoding="utf-8"))
    after = parse(coverage.BOUNDS_DOC.read_text(encoding="utf-8"))
    after["Nystroem"]["hashes"]["s"] = "changed"
    assert relaxations(before, after) == ["Nystroem: K/S/tau/map source hashes changed"]


def test_the_bound_guard_catches_a_function_raise_below_the_class_ceiling():
    from scripts.check_bounds import parse, relaxations

    before = parse(coverage.BOUNDS_DOC.read_text(encoding="utf-8"))
    after = parse(coverage.BOUNDS_DOC.read_text(encoding="utf-8"))
    after["FunctionTransformer"]["functions"]["exp"] += 1
    assert relaxations(before, after) == [
        "FunctionTransformer: function ulps exp: 1 -> 2"
    ]


def test_the_bound_guard_lists_ceiling_and_shape_k_increases():
    from scripts.check_bounds import parse, relaxations

    before = parse(coverage.BOUNDS_DOC.read_text(encoding="utf-8"))
    after = parse(coverage.BOUNDS_DOC.read_text(encoding="utf-8"))
    after["FunctionTransformer"]["ulps"] += 1
    key = next(iter(after["Nystroem"]["ks"]))
    after["Nystroem"]["ks"][key] += 1
    findings = relaxations(before, after)
    assert "FunctionTransformer: ulp ceiling 3 -> 4" in findings
    assert any(f"Nystroem: K {key}:" in finding for finding in findings)


def test_the_bound_guard_allows_unchanged_lower_and_new_bounds():
    from scripts.check_bounds import parse, relaxations

    before = parse(coverage.BOUNDS_DOC.read_text(encoding="utf-8"))
    after = parse(coverage.BOUNDS_DOC.read_text(encoding="utf-8"))
    assert relaxations(before, after) == []
    after["FunctionTransformer"]["ulps"] -= 1
    after["FunctionTransformer"]["functions"]["cbrt"] -= 1
    after["NewClass"] = after["Nystroem"]
    assert relaxations(before, after) == []


def test_the_bound_guard_checks_committed_changes_and_requires_a_closed_decision(
    tmp_path,
):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from scripts.check_bounds import MANIFEST

    root = Path(__file__).resolve().parents[4]
    script = tmp_path / "scripts" / "check_bounds.py"
    script.parent.mkdir()
    shutil.copyfile(root / "scripts" / "check_bounds.py", script)
    manifest = tmp_path / MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(coverage.bounds_table(), encoding="utf-8")

    def git(*args):
        return subprocess.check_output(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=tmp_path,
            text=True,
        ).strip()

    def commit():
        git("add", ".")
        git(
            "-c",
            "user.name=Bound Test",
            "-c",
            "user.email=bounds@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "synthetic bound change",
        )

    def check():
        return subprocess.run(  # noqa: S603
            [sys.executable, str(script), base],
            cwd=tmp_path,
            text=True,
            capture_output=True,
        )

    git("init", "-q")
    commit()
    base = git("rev-parse", "HEAD")
    (tmp_path / "unrelated.txt").write_text("unrelated change")
    commit()
    assert check().returncode == 0
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('"cbrt": 3', '"cbrt": 4'),
        encoding="utf-8",
    )
    commit()
    result = check()
    assert result.returncode == 1
    assert "FunctionTransformer: function ulps cbrt: 3 -> 4" in result.stdout
    decision = tmp_path / "proposals/closed/bound-change.md"
    decision.parent.mkdir(parents=True)
    decision.write_text("Synthetic closed proposal for this regression.")
    commit()
    assert check().returncode == 0


def test_writing_the_coverage_table_keeps_utf8_text(tmp_path, monkeypatch):
    doc = tmp_path / "coverage.md"
    text = "Before: µ\n" + coverage.table() + "\nAfter: τ\n"
    doc.write_bytes(text.encode("utf-8"))
    monkeypatch.setattr(coverage, "DOC", doc)
    coverage.write()
    assert doc.read_text(encoding="utf-8") == text
