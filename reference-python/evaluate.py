"""Optional reference CLI. Run from an environment with NumPy installed."""
import argparse
import json
from scoring import score
from paired_bootstrap import compare, load, load_gold

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["score", "compare"])
parser.add_argument("--gold", required=True)
parser.add_argument("--pred")
parser.add_argument("--a")
parser.add_argument("--b")
parser.add_argument("--resamples", type=int, default=50000)
parser.add_argument("--family-size", type=int)
args = parser.parse_args()
gold = load_gold(args.gold)
# Gold targets may be objects in the synthetic fixture or JSON strings in legacy data.
for row in gold:
    if isinstance(row["target"], dict):
        row["target"] = json.dumps(row["target"])
if args.mode == "score":
    if not args.pred:
        parser.error("score requires --pred")
    predictions = load(args.pred)
    if set(predictions) - {row["id"] for row in gold}:
        parser.error("predictions contain IDs outside gold")
    with open(args.pred, encoding="utf-8") as source:
        truncation = {row["id"]: bool(row.get("truncated"))
                      for row in (json.loads(line) for line in source)}
    result = score(gold, predictions, truncation)
else:
    if not args.a or not args.b:
        parser.error("compare requires --a and --b")
    result = compare(gold, load(args.a), load(args.b), resamples=args.resamples,
                     ni_family_size=args.family_size)
print(json.dumps(result, indent=2))
