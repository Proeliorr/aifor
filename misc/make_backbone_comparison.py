"""Rebuild `docs/wandb_backbone_comparison.{md,html}` from the two pooled reports.

Parses the `evaluation_total.txt` written by `evaluation_stats_FOR.py` for each
backbone, pairs their report blocks by date, and emits two files: a
GitHub-flavoured-markdown table set that pastes into a Weights & Biases report,
and a styled HTML version for a W&B media panel (see `log_html_to_wandb.py`).

**Every number comes from the report files -- nothing is retyped.** That is the
whole point of the script: it exists so the comparison tables in `docs/` can be
regenerated and checked against their source rather than trusted.

Run it from the repository root::

    python misc/make_backbone_comparison.py

The two run directories below are the evaluations the report quotes. Both were
produced in the same cache regime, which is what makes them comparable -- see
`docs/eval_process.md` section 11 before pointing this at different runs.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "ForAINet" / "pre-trained_models" / "eval"
RUNS = {
    "mink": EVAL / "PointGroup-PAPER" / "2026-09-17_13-43-04" / "evaluation_total.txt",
    "ts": EVAL / "PointGroup-PAPER-TS" / "2026-09-16_18-03-54" / "evaluation_total.txt",
}

CLASS_NAMES = ["unclassified", "low_vegetation", "ground", "stem_points", "live_branches"]


def parse(path: Path) -> list[dict]:
    """Split a report file into blocks and turn each into {metric: value}."""
    blocks = []
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("pooled report"):
            cur = {"_date": line.split()[-2], "_plots": []}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s*\[\d+\]\s+(\S+)", line)
        if m:
            cur["_plots"].append(m.group(1))
            continue
        m = re.match(r"(.+?):\s+(.+)$", line)
        if not m or line.startswith(("  run dir", "  pooling")):
            continue
        key, raw = m.group(1).strip(), m.group(2).strip()
        if raw.startswith("["):
            # printed either as a python list ("[0.0, 0.71, ...]") or as a numpy
            # repr ("[0.71 0.69 0.90]") depending on the metric
            cur[key] = [float(v) for v in raw.strip("[]").replace(",", " ").split()]
        else:
            try:
                cur[key] = float(raw)
            except ValueError:
                pass
    return blocks


def val(block, key, idx=None):
    v = block[key]
    return v[idx] if idx is not None else v


# (section, label, key, index-or-None)
ROWS = [
    ("Semantic segmentation (4 classes)", "Overall accuracy", "Semantic Segmentation oAcc", None),
    ("Semantic segmentation (4 classes)", "Mean accuracy", "Semantic Segmentation mAcc", None),
    ("Semantic segmentation (4 classes)", "**mIoU**", "Semantic Segmentation mIoU", None),
    ("Semantic segmentation (4 classes)", "Overall accuracy, ground excluded", "Semantic Segmentation oAcc without ground points", None),
    ("Semantic segmentation (4 classes)", "Mean accuracy, ground excluded", "Semantic Segmentation mAcc without ground points", None),
    ("Semantic segmentation (4 classes)", "**mIoU, ground excluded**", "Semantic Segmentation mIoU without ground points", None),

    ("Per-class IoU", "1 low_vegetation", "Semantic Segmentation IoU", 1),
    ("Per-class IoU", "2 ground", "Semantic Segmentation IoU", 2),
    ("Per-class IoU", "3 stem_points *(thing)*", "Semantic Segmentation IoU", 3),
    ("Per-class IoU", "4 live_branches *(thing)*", "Semantic Segmentation IoU", 4),

    ("Binary segmentation (tree / not-tree)", "Overall accuracy", "Binary Semantic Segmentation oAcc", None),
    ("Binary segmentation (tree / not-tree)", "Mean accuracy", "Binary Semantic Segmentation mAcc", None),
    ("Binary segmentation (tree / not-tree)", "IoU, not-tree", "Binary Semantic Segmentation IoU", 1),
    ("Binary segmentation (tree / not-tree)", "IoU, tree", "Binary Semantic Segmentation IoU", 2),
    ("Binary segmentation (tree / not-tree)", "**mIoU**", "Binary Semantic Segmentation mIoU", None),

    ("Instance segmentation (individual trees)", "MUCov", "Instance Segmentation mMUCov", None),
    ("Instance segmentation (individual trees)", "MWCov", "Instance Segmentation mMWCov", None),
    ("Instance segmentation (individual trees)", "Precision", "Instance Segmentation mPrecision", None),
    ("Instance segmentation (individual trees)", "Recall", "Instance Segmentation mRecall", None),
    ("Instance segmentation (individual trees)", "**F1**", "Instance Segmentation F1 score", None),

    ("Panoptic quality", "RQ (things)", "Instance Segmentation meanRQ (things)", None),
    ("Panoptic quality", "SQ (things)", "Instance Segmentation meanSQ (things)", None),
    ("Panoptic quality", "**PQ (things)**", "Instance Segmentation meanPQ (things)", None),
    ("Panoptic quality", "RQ (stuff)", "Instance Segmentation meanRQ (stuff)", None),
    ("Panoptic quality", "SQ (stuff)", "Instance Segmentation meanSQ (stuff)", None),
    ("Panoptic quality", "PQ (stuff)", "Instance Segmentation meanPQ (stuff)", None),
    ("Panoptic quality", "mean PQ (all)", "Instance Segmentation meanPQ", None),
]

TIE = 5e-5  # below this the 4-decimal display would show the same number


def rows_for(a, b):
    """Yield (section, label, mink, ts, delta, winner) for one pair of blocks."""
    for section, label, key, idx in ROWS:
        x, y = val(a, key, idx), val(b, key, idx)
        d = x - y
        winner = "tie" if abs(d) < TIE else ("mink" if d > 0 else "ts")
        yield section, label, x, y, d, winner


def fmt(v):
    return f"{v:.4f}"


def fmt_d(d):
    return "0" if abs(d) < TIE else f"{d:+.4f}"


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------
def md_group(title, subtitle, a, b):
    out = [f"### {title}", "", subtitle, ""]
    section = None
    for sec, label, x, y, d, w in rows_for(a, b):
        if sec != section:
            section = sec
            out += ["", f"**{sec}**", "",
                    "| Metric | MinkowskiEngine | TorchSparse 1.4 | Δ (Mink − TS) |",
                    "|---|---:|---:|---:|"]
        mx, my = fmt(x), fmt(y)
        if w == "mink":
            mx = f"**{mx}** 🟢"
        elif w == "ts":
            my = f"**{my}** 🟠"
        out.append(f"| {label} | {mx} | {my} | {fmt_d(d)} |")
    return "\n".join(out)


def tally(a, b):
    c = {"mink": 0, "ts": 0, "tie": 0}
    for *_, w in rows_for(a, b):
        c[w] += 1
    return c


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------
CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#6a737d; --line:#e3e6ea;
        --win:#16794a; --winbg:#e8f6ef; --lose:#b45309; --losebg:#fdf3e3; }
body { background:var(--bg); color:var(--fg); margin:0; padding:20px;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       font-size:14px; line-height:1.5; }
h2 { margin:0 0 4px; font-size:20px; }
h3 { margin:28px 0 2px; font-size:16px; }
p.sub { margin:0 0 14px; color:var(--muted); font-size:13px; }
table { border-collapse:collapse; width:100%; max-width:820px; margin:10px 0 18px; }
caption { caption-side:top; text-align:left; font-weight:600; padding:10px 0 6px; font-size:13px;
          letter-spacing:.02em; text-transform:uppercase; color:var(--muted); }
th,td { border-bottom:1px solid var(--line); padding:7px 10px; text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead th { background:#f5f7f9; border-bottom:2px solid var(--line); font-weight:600; white-space:nowrap; }
tbody tr:hover { background:#fafbfc; }
td.win  { background:var(--winbg);  color:var(--win);  font-weight:700; border-radius:3px; }
td.lose { background:var(--losebg); color:var(--lose); font-weight:700; border-radius:3px; }
td.num { font-variant-numeric:tabular-nums; }
td.d-pos { color:var(--win); } td.d-neg { color:var(--lose); }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px;
         font-weight:600; background:#eef2f6; color:#33415c; margin-right:6px; }
.note { max-width:820px; border-left:3px solid #c8cdd3; background:#f7f9fa; padding:10px 14px;
        margin:16px 0; color:#444; font-size:13px; }
"""


def html_group(title, subtitle, a, b):
    out = [f"<h3>{title}</h3>", f'<p class="sub">{subtitle}</p>']
    section, body = None, []
    for sec, label, x, y, d, w in rows_for(a, b):
        if sec != section:
            if section:
                body.append("</tbody></table>")
                out += body
                body = []
            section = sec
            body.append(
                f'<table><caption>{sec}</caption><thead><tr>'
                f'<th>Metric</th><th>MinkowskiEngine</th><th>TorchSparse 1.4</th>'
                f'<th>&Delta; (Mink &minus; TS)</th></tr></thead><tbody>'
            )
        lbl = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", label)
        lbl = re.sub(r"\*(.+?)\*", r"<em>\1</em>", lbl)
        cx = ' class="num win"' if w == "mink" else ' class="num"'
        cy = ' class="num win"' if w == "ts" else ' class="num"'
        cd = "num d-pos" if d > TIE else ("num d-neg" if d < -TIE else "num")
        body.append(
            f"<tr><td>{lbl}</td><td{cx}>{fmt(x)}</td><td{cy}>{fmt(y)}</td>"
            f'<td class="{cd}">{fmt_d(d)}</td></tr>'
        )
    body.append("</tbody></table>")
    out += body
    return "\n".join(out)


def main():
    blocks = {k: parse(p) for k, p in RUNS.items()}
    # block 0 = 2026-09-17 (two test plots), block 1 = 2026-09-18 (all three)
    groups = [
        ("Group A — held-out test plots (17 Sep)",
         "plot_01_test + plot_14_test, pooled. Neither plot was seen in training or validation, "
         "so this is the comparison to quote.",
         0),
        ("Group B — all three evaluation plots (18 Sep)",
         "plot_01_test + plot_14_test + plot_11_val, pooled. plot_11 was training's validation plot, "
         "so these numbers are not a clean held-out measure — they are here for completeness.",
         1),
    ]

    intro_md = [
        "## MinkowskiEngine vs TorchSparse 1.4",
        "",
        "Same ForAINet / PointGroup model, same training data, same 4-class scheme "
        "(`low_vegetation`, `ground`, `stem_points`, `live_branches`); the **only** difference is "
        "the sparse-convolution backbone. Every figure below is a *pooled* score — the confusion "
        "matrix and the instance matches are accumulated across plots and the metric is computed "
        "once, which is not the same as averaging the per-plot reports.",
        "",
        "🟢 = MinkowskiEngine ahead 🟠 = TorchSparse ahead. Higher is better for every metric shown.",
    ]
    intro_html = [
        "<h2>MinkowskiEngine vs TorchSparse 1.4</h2>",
        '<p class="sub">Same ForAINet / PointGroup model, same training data, same 4-class scheme '
        "(low_vegetation, ground, stem_points, live_branches); the <strong>only</strong> difference "
        "is the sparse-convolution backbone. Every figure is a <em>pooled</em> score — the confusion "
        "matrix and the instance matches are accumulated across plots and the metric computed once, "
        "which is not the same as averaging the per-plot reports.</p>",
        '<p><span class="badge">green = MinkowskiEngine ahead</span>'
        '<span class="badge">orange = TorchSparse ahead</span>'
        "Higher is better for every metric shown.</p>",
    ]

    md, html = list(intro_md), list(intro_html)
    for title, subtitle, i in groups:
        a, b = blocks["mink"][i], blocks["ts"][i]
        assert a["_plots"] == b["_plots"], (a["_plots"], b["_plots"])
        c = tally(a, b)
        summary = (f"MinkowskiEngine leads on {c['mink']} of the {sum(c.values())} metrics, "
                   f"TorchSparse on {c['ts']}" + (f", {c['tie']} tied." if c["tie"] else "."))
        md += ["", md_group(title, subtitle + f" **{summary}**", a, b)]
        html += [html_group(title, subtitle + f" <strong>{summary}</strong>", a, b)]

    outro_md = [
        "",
        "### Reading the numbers",
        "",
        "- **Semantic segmentation is close.** Roughly one point of mIoU separates the backbones on "
        "the held-out plots — both learned essentially the same point classifier.",
        "- **Instance segmentation is where they part.** MinkowskiEngine recovers noticeably more "
        "trees (recall, F1, coverage), which is what drives the PQ (things) gap.",
        "- **The only instance metric TorchSparse wins is SQ (things)** — segmentation *quality* of "
        "the instances it did find. It matches fewer trees, but the ones it matches overlap the "
        "ground truth slightly better. RQ × SQ = PQ, and the RQ gap dominates. (In Group B it also "
        "edges out `ground` IoU.)",
        "- **RQ (stuff) is 1.0 for both** by construction: there is a single stuff region per plot "
        "and both models find it, so the stuff column really only reports its IoU.",
        "- **These differences are not evaluation noise.** Re-running the same checkpoint through "
        "the same evaluation path reproduced the numbers exactly. They are, however, one *training* "
        "run per backbone, so what is compared is these two trained models — not the two libraries "
        "in general.",
    ]
    outro_html = [
        "<h3>Reading the numbers</h3>",
        '<div class="note"><ul>'
        "<li><strong>Semantic segmentation is close.</strong> Roughly one point of mIoU separates "
        "the backbones on the held-out plots — both learned essentially the same point classifier.</li>"
        "<li><strong>Instance segmentation is where they part.</strong> MinkowskiEngine recovers "
        "noticeably more trees (recall, F1, coverage), which drives the PQ (things) gap.</li>"
        "<li><strong>The only instance metric TorchSparse wins is SQ (things)</strong> — segmentation "
        "<em>quality</em> of the instances it did find. It matches fewer trees, but the ones it "
        "matches overlap the ground truth slightly better. RQ &times; SQ = PQ, and the RQ gap "
        "dominates. (In Group B it also edges out <code>ground</code> IoU.)</li>"
        "<li><strong>RQ (stuff) is 1.0 for both</strong> by construction: one stuff region per plot, "
        "found by both, so the stuff column really only reports its IoU.</li>"
        "<li><strong>These differences are not evaluation noise.</strong> Re-running the same "
        "checkpoint through the same evaluation path reproduced the numbers exactly. They are, "
        "however, one <em>training</em> run per backbone, so what is compared is these two trained "
        "models — not the two libraries in general.</li>"
        "</ul></div>",
    ]
    md += outro_md
    html += outro_html

    docs = ROOT / "docs"
    (docs / "wandb_backbone_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (docs / "wandb_backbone_comparison.html").write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n<style>"
        + CSS + "</style></head>\n<body>\n" + "\n".join(html) + "\n</body></html>\n",
        encoding="utf-8",
    )
    for k, bl in blocks.items():
        print(k, [(b["_date"], len(b["_plots"])) for b in bl])
    print("wrote docs/wandb_backbone_comparison.{md,html}")


if __name__ == "__main__":
    main()
