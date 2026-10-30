#!/usr/bin/env python3
"""Every bench run on one page: what was set, and held-out laps per seed
with the mean and its 95% bootstrap interval, newest first. A setting swept
on its own at three or more values (n_rays=4, n_rays=8, n_rays=16) also gets
a chart: held-out laps against the setting, with the intervals.

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


ONE_SETTING = re.compile(r"^(?P<key>[a-z_0-9]+)=(?P<val>-?\d+(?:\.\d+)?)$")


def sweeps(runs):
    """{setting: [runs]} for every setting that was swept on its own: logs whose
    whole settings string is `key=number`, at three or more different numbers,
    the newest log for each number, ordered by number. Runs with an exam
    change, or several settings at once, are not part of a sweep."""
    by = {}
    for r in sorted(runs, key=lambda r: r["mtime"]):  # oldest first: newer overwrite
        if m := ONE_SETTING.match(r["set"]):
            by.setdefault(m["key"], {})[float(m["val"])] = r
    return {k: [v[x] for x in sorted(v)] for k, v in by.items() if len(v) >= 3}


def chart(key, runs, w=520, h=220):
    """Held-out laps (mean and 95% bootstrap interval over seeds) against one
    swept setting, as an inline SVG. The y axis is the whole exam, 0 to all laps."""
    of = runs[0]["held-out"]["of"]
    xs = [float(ONE_SETTING.match(r["set"])["val"]) for r in runs]
    x0, x1, pad = min(xs), max(xs), 36
    px = lambda v: pad + (w - 2 * pad) * ((v - x0) / (x1 - x0) if x1 > x0 else 0.5)
    py = lambda v: h - pad + (2 * pad - h + 8) * v / of
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="held-out laps against {html.escape(key)}">',
           f'<line x1="{pad}" y1="{py(0):.1f}" x2="{w - pad}" y2="{py(0):.1f}" stroke="#232733"/>',
           f'<text x="4" y="{py(of) + 4:.0f}" fill="#8b93a7" font-size="10">{of}</text><text x="4" y="{py(0) + 4:.0f}" fill="#8b93a7" font-size="10">0</text>']
    pts = []
    for r, v in zip(runs, xs):
        x = px(v)
        out.append(f'<line x1="{x:.1f}" y1="{py(r["lo"]):.1f}" x2="{x:.1f}" y2="{py(r["hi"]):.1f}" stroke="#6ea8ff" stroke-width="3" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{py(r["mean"]):.1f}" r="3.5" fill="#e6e8ee"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 8}" fill="#8b93a7" font-size="10" text-anchor="middle">{v:g}</text>')
        pts.append(f"{x:.1f},{py(r['mean']):.1f}")
    out.insert(2, f'<polyline points="{" ".join(pts)}" fill="none" stroke="#8b93a7" stroke-dasharray="3 3"/>')
    return "".join(out) + "</svg>"


def page(runs):
    charts = "".join(f'<h2>{html.escape(k)}</h2>{chart(k, rs)}' for k, rs in sweeps(runs).items())
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
h1 {{ font-size:16px; }} h2 {{ font-size:13px; margin:18px 0 4px; }} p {{ color:var(--dim); max-width:80ch; }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; min-width:860px; }}
td, th {{ padding:5px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ color:var(--dim); font-weight:normal; }} .n {{ text-align:right; font-variant-numeric:tabular-nums; }} .f {{ color:var(--dim); }}
.bar {{ position:relative; width:200px; }} .ci {{ position:absolute; top:9px; height:6px; border-radius:3px; background:var(--blue); }}
.pt {{ position:absolute; top:7px; width:2px; height:10px; background:var(--ink); }}
</style></head><body><h1>Bench runs, newest first</h1>
<p>Held-out laps: the best generalist of each seed's final population, on tracks it never trained on. The interval is a 95% bootstrap over seeds; the bar spans the whole exam, 0 to all laps.</p>
{charts}<div class="wrap"><table><tr><th>log</th><th>settings</th><th>pop</th><th>gens</th><th>seeds</th><th>held-out</th><th>95%</th><th></th><th>lap ticks</th></tr>
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
