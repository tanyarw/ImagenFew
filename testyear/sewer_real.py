"""One real year under BC and EFD, from empty tanks: flood-control's own `bc_efd_check.one`, the function behind its
saved real rows, so 2009 is simulated exactly as 2000-2008 were. Runs in flood-control's environment (pyswmm); the
test starts it (run.py), so it is not run by hand:

    cd ../flood-control && PYTHONPATH=.:../ImagenFew .venv/bin/python -m testyear.sewer_real --year 2008 \\
        --prereg-sha256 <sha> --out ../ImagenFew/results/testyear/sewer_real_2008.json

The test year also needs the claim file that guard.unlock writes, with the same preregistration hash: this module
cannot open 2009 on its own.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import config as K
from .guard import Locked


def _one(job):
    from analysis.synthetic_rain.bc_efd_check import one
    r = one(job)
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}      # numpy scalars -> JSON


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--prereg-sha256", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.year == K.TEST_YEAR:
        c = K.claim_path(K.TEST_YEAR)
        if not c.exists() or json.loads(c.read_text()).get("prereg_sha256") != a.prereg_sha256:
            raise Locked(f"no claim for this preregistration ({c.name}); the test opens the year, not this module")
    elif a.year not in (K.REHEARSAL_YEAR, *K.TRAIN_YEARS):
        raise SystemExit(f"{a.year}: not a year of the record this test uses")
    with ProcessPoolExecutor(len(K.CONTROLLERS)) as pool:
        rows = list(pool.map(_one, [("real", a.year, c) for c in K.CONTROLLERS]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"year": a.year, "prereg_sha256": a.prereg_sha256, "rows": rows,
                               "function": "flood-control analysis/synthetic_rain/bc_efd_check.one"}, indent=1) + "\n")
    tmp.replace(out)
    for r in rows:
        print(f"real {r['year']} {r['controller']:3s} {r['total_m3']:10.0f} m3", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
