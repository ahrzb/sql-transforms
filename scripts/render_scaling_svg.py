"""Render benchmarks/scaling_results.json as a dependency-free SVG chart
(log-log: batch size vs ns/row) — one panel per scenario, four series.

    .venv/Scripts/python scripts/render_scaling_svg.py > out.svg-fragment

Colors ride CSS variables so the report artifact themes them; a fallback
palette is inlined for standalone viewing.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SERIES = [
    ("columnar", "var(--c-columnar, #2563eb)"),
    ("row", "var(--c-row, #7c3aed)"),
    ("duckdb", "var(--c-duckdb, #d97706)"),
    ("python", "var(--c-python, #059669)"),
]
W, H, PAD_L, PAD_B, PAD_T, PAD_R = 360, 260, 46, 34, 22, 10


def panel(res: dict, x0: float, y0: float) -> str:
    pts = res["points"]
    xs = [p["n"] for p in pts]
    # per-ROW cost: total / n.
    all_y = [p[k] / p["n"] for p in pts for k, _ in SERIES]
    lx0, lx1 = math.log10(min(xs)), math.log10(max(xs))
    ly0, ly1 = math.log10(min(all_y)) - 0.05, math.log10(max(all_y)) + 0.05
    iw, ih = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(n):
        return x0 + PAD_L + (math.log10(n) - lx0) / (lx1 - lx0) * iw

    def sy(v):
        return y0 + PAD_T + (1 - (math.log10(v) - ly0) / (ly1 - ly0)) * ih

    s = [
        '<g font-family="inherit" font-size="10">',
        f'<text x="{x0 + PAD_L}" y="{y0 + 13}" font-size="12" font-weight="600" '
        f'fill="var(--ink, #111)">{res["scenario"]}</text>',
    ]
    # Gridlines: decades on both axes.
    for d in range(int(math.ceil(lx0)), int(lx1) + 1):
        gx = sx(10**d)
        s.append(
            f'<line x1="{gx:.1f}" y1="{y0 + PAD_T}" x2="{gx:.1f}" '
            f'y2="{y0 + H - PAD_B}" stroke="var(--grid, #0001)" stroke-width="1"/>'
        )
        s.append(
            f'<text x="{gx:.1f}" y="{y0 + H - PAD_B + 14}" text-anchor="middle" '
            f'fill="var(--ink-3, #888)">{10**d:g}</text>'
        )
    for d in range(int(math.ceil(ly0)), int(ly1) + 1):
        gy = sy(10**d)
        lbl = (
            f"{10**d:g}"
            if d < 3
            else f"{10 ** (d - 3):g}µ"
            if d < 6
            else f"{10 ** (d - 6):g}m"
        )
        s.append(
            f'<line x1="{x0 + PAD_L}" y1="{gy:.1f}" x2="{x0 + W - PAD_R}" '
            f'y2="{gy:.1f}" stroke="var(--grid, #0001)" stroke-width="1"/>'
        )
        s.append(
            f'<text x="{x0 + PAD_L - 5}" y="{gy + 3:.1f}" text-anchor="end" '
            f'fill="var(--ink-3, #888)">{lbl}s</text>'
        )
    for key, color in SERIES:
        d = " ".join(
            f"{'M' if i == 0 else 'L'} {sx(p['n']):.1f} {sy(p[key] / p['n']):.1f}"
            for i, p in enumerate(pts)
        )
        s.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round"/>'
        )
        last = pts[-1]
        s.append(
            f'<circle cx="{sx(last["n"]):.1f}" cy="{sy(last[key] / last["n"]):.1f}" '
            f'r="3" fill="{color}"/>'
        )
    s.append("</g>")
    return "\n".join(s)


def main():
    src = Path(__file__).parent.parent / "benchmarks" / "scaling_results.json"
    results = json.loads(src.read_text(encoding="utf-8"))
    cols = 2
    rows = (len(results) + 1) // cols
    tw, th = cols * (W + 14), rows * (H + 12) + 26
    out = [
        f'<svg viewBox="0 0 {tw} {th}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Batch size vs ns per row, log-log">'
    ]
    for i, res in enumerate(results):
        out.append(panel(res, (i % cols) * (W + 14), (i // cols) * (H + 12)))
    # Legend.
    lx = 10
    for key, color in SERIES:
        out.append(
            f'<rect x="{lx}" y="{th - 18}" width="10" height="10" '
            f'fill="{color}" rx="2"/>'
            f'<text x="{lx + 14}" y="{th - 9}" font-size="11" '
            f'fill="var(--ink, #111)">{key}</text>'
        )
        lx += 90
    out.append(
        f'<text x="{tw - 8}" y="{th - 9}" font-size="10" text-anchor="end" '
        f'fill="var(--ink-3, #888)">x: rows/call · y: time PER ROW (log-log)</text>'
    )
    out.append("</svg>")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
