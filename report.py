#!/usr/bin/env python3
"""Every bench run on one page: what was set, and held-out laps per seed
with the mean and its 95% bootstrap interval, newest first.

    python report.py                 # reads tmp/*.log, writes tmp/report.html
    python report.py logs/ out.html
"""
import glob
import html
import os
import re
import sys

from bench import bootstrap

HEADER = re.compile(r"^(?P<set>.*?)\s+pop=(?P<pop>\d+) gens=(?P<gens>\d+) seeds=(?P<seeds>\d+)")
PER_SEED = re.compile(r"^\s*(?P<which>train|held-out)\s+laps/(?P<of>\d+) per seed:\s*(?P<laps>[\d ]+)$")
TICKS = re.compile(r"^\s*(?P<which>train|held-out)\s+laps mean .*lap ticks mean (?P<ticks>\d+)")


def parse(path):
    """One bench log -> a dict, or None if it is not a finished bench run."""
    run = {"file": os.path.basename(path), "mtime": os.path.getmtime(path)}
    with open(path) as f:
        for line in f:
            if (m := HEADER.match(line)) and "pop" not in run:
                run.update(set=m["set"].strip(), pop=int(m["pop"]), gens=int(m["gens"]), seeds=int(m["seeds"]))
            elif m := PER_SEED.match(line):
                run[m["which"]] = {"of": int(m["of"]), "laps": [int(x) for x in m["laps"].split()]}
            elif m := TICKS.match(line):
                run.setdefault(m["which"], {})["ticks"] = int(m["ticks"])
    if "pop" not in run or "laps" not in run.get("held-out", {}):
        return None
    laps = run["held-out"]["laps"]
    run["mean"] = sum(laps) / len(laps)
    run["lo"], run["hi"] = bootstrap(laps)
    return run


def page(runs):
    rows = []
    for r in runs:
        h, of = r["held-out"], r["held-out"]["of"]
        x = lambda v: f"{100 * v / of:.1f}%"
        rows.append(f"""<tr><td class="f">{html.escape(r['file'])}</td><td>{html.escape(r['set'])}</td>
<td>{r['pop']}</td><td>{r['gens']}</td><td>{r['seeds']}</td>
<td class="n">{r['mean']:.2f}/{of}</td><td class="n">{r['lo']:.2f}-{r['hi']:.2f}</td>
<td class="bar"><span class="ci" style="left:{x(r['lo'])};width:calc({x(r['hi'])} - {x(r['lo'])})"></span><span class="pt" style="left:{x(r['mean'])}"></span></td>
<td class="n">{h.get('ticks', '')}</td></tr>""")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cars: bench runs</title><style>
:root {{ --bg:#0e0f13; --panel:#14161c; --line:#232733; --ink:#e6e8ee; --dim:#8b93a7; --blue:#6ea8ff; }}
body {{ margin:0; padding:20px; background:var(--bg); color:var(--ink); font:13px/1.5 ui-monospace,monospace; }}
h1 {{ font-size:16px; }} p {{ color:var(--dim); max-width:80ch; }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; min-width:860px; }}
td, th {{ padding:5px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ color:var(--dim); font-weight:normal; }} .n {{ text-align:right; font-variant-numeric:tabular-nums; }} .f {{ color:var(--dim); }}
.bar {{ position:relative; width:200px; }} .ci {{ position:absolute; top:9px; height:6px; border-radius:3px; background:var(--blue); }}
.pt {{ position:absolute; top:7px; width:2px; height:10px; background:var(--ink); }}
</style></head><body><h1>Bench runs, newest first</h1>
<p>Held-out laps: the best generalist of each seed's final population, on tracks it never trained on. The interval is a 95% bootstrap over seeds; the bar spans the whole exam, 0 to all laps.</p>
<div class="wrap"><table><tr><th>log</th><th>settings</th><th>pop</th><th>gens</th><th>seeds</th><th>held-out</th><th>95%</th><th></th><th>lap ticks</th></tr>
{''.join(rows)}</table></div></body></html>"""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "tmp"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("tmp", "report.html")
    runs = [r for r in (parse(p) for p in glob.glob(os.path.join(src, "*.log"))) if r]
    runs.sort(key=lambda r: -r["mtime"])
    with open(out, "w") as f:
        f.write(page(runs))
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
