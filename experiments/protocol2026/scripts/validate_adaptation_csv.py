import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
model = sys.argv[2]
family = sys.argv[3]
seed = int(sys.argv[4])

expected = {
    (regime, step)
    for regime in ("fine_tune", "from_scratch")
    for step in (10, 100, 1000)
}

try:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1200

    groups = {}
    for row in rows:
        assert row["model"] == model
        assert row["train_family"] == family
        assert int(row["train_seed"]) == seed

        groups.setdefault(row["path"], []).append(
            (row["regime"], int(row["steps"]))
        )

    assert len(groups) == 200
    assert all(
        len(v) == 6 and set(v) == expected
        for v in groups.values()
    )

    print("VALID", path)
except Exception as e:
    print("INVALID", path, e)
    sys.exit(1)
