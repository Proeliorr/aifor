"""Log a local HTML file to Weights & Biases as a media panel.

Why this exists
---------------
W&B *Report* markdown blocks strip raw HTML, so inline CSS never survives a
copy-paste.  The only place arbitrary HTML renders with its own styling is a
**media panel**: `wandb.Html` stores the file as run media, and the report panel
shows it inside a sandboxed iframe where our `<style>` block applies in full.

So the flow is: log the HTML to a run here -> add that run's media panel to the
report in the browser.

Typical use (from the repo root, in the `aifor` env)::

    python misc/log_html_to_wandb.py docs/wandb_backbone_comparison.html

which creates a small dedicated run named ``report-assets`` in
``aifor/FOR-dataset`` holding one media key, ``backbone_comparison``.

To attach the panel to an existing run instead of making a new one, pass that
run's id (the short code in its URL)::

    python misc/log_html_to_wandb.py docs/wandb_backbone_comparison.html --resume-id ab12cd34

Nothing here touches training: the run it creates logs no metrics, so it will
not appear in loss/accuracy charts.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("html", type=Path, help="the .html file to upload")
    p.add_argument("--entity", default="aifor", help="wandb entity (default: %(default)s)")
    p.add_argument("--project", default="FOR-dataset", help="wandb project (default: %(default)s)")
    p.add_argument("--run-name", default="report-assets",
                   help="name of the run to create (default: %(default)s)")
    p.add_argument("--key", default="backbone_comparison",
                   help="media key the panel will be listed under (default: %(default)s)")
    p.add_argument("--resume-id", default=None,
                   help="attach to this existing run id instead of creating a new run")
    args = p.parse_args()

    if not args.html.is_file():
        raise SystemExit(f"no such file: {args.html}")

    # Imported late so that --help works without wandb installed.
    import wandb

    html = args.html.read_text(encoding="utf-8")
    print(f"{args.html} -> {len(html):,} bytes")

    init = dict(entity=args.entity, project=args.project, job_type="report-asset")
    if args.resume_id:
        init.update(id=args.resume_id, resume="must")
    else:
        init.update(name=args.run_name)

    run = wandb.init(**init)
    # inject=False keeps the document byte-for-byte; the default (True) prepends
    # wandb's own stylesheet, which would fight with the one in our <head>.
    run.log({args.key: wandb.Html(html, inject=False)})
    url = run.url
    run.finish()

    print(f"\nlogged as media key '{args.key}'")
    print(f"run: {url}")
    print("Now in your report: Add panel -> Media -> pick this run and that key.")


if __name__ == "__main__":
    main()
