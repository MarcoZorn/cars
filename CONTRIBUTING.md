# Working on cars

Python 3 with numpy and pygame (`requirements.txt`).

## Tests

```
python test_cars.py          # or: python -m pytest -q test_cars.py
```

Every new behaviour comes with a test in `test_cars.py`.

## Where results go

- A benchmark run is `python bench.py --seeds 8 --pop 200 --gens 20 --set <field>=<value>`; redirect its
  output to a log, `tmp/<name>.log` (`tmp/` and `*.log` are not committed).
- `python report.py` turns every log in `tmp/` into `tmp/report.html`: held-out laps per seed, the mean
  with a 95% bootstrap interval, and a chart for any setting swept alone at three or more values.
- What is worth keeping goes into `README.md` with its numbers, copied from the log, and the settings that
  produced them.

## Rules

- Compare settings on held-out tracks and across seeds, with the interval, not on one seed's training laps.
- A change that did not help is written up as plainly as one that did.
- Every number you write is copied from a command's output, not typed from memory.

## Commits

Small, one idea each, terse messages. Check `git status` before committing.
