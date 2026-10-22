"""Command line: `python -m testyear <command>` from the repository root (README.md has the order).

  status                        what exists: preregistration, rehearsal, claim, result; whether all is committed
  register [--series NAME]      fix the generator's 40 years and the rule -> testyear/preregistration.json
  rehearse                      the whole test on 2008 -> results/testyear/rehearsal_2008.json
  test --spend-test-year        the test on 2009, once -> results/testyear/test_2009.json
"""
import argparse
import datetime as dt
import sys

from sitetransfer.provenance import ROOT, RunRecord, rel

from . import config as K


def _record(command, params):
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RunRecord(command, "astlingen", params, ROOT / "results" / "testyear" / "records" / f"{stamp}_{command}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m testyear", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "register", "rehearse", "test"])
    ap.add_argument("--series", default=K.SERIES, help="register: the generated series (members _m1.. follow)")
    ap.add_argument("--spend-test-year", action="store_true", help="test: open 2009, once")
    a = ap.parse_args(argv)
    if a.command == "test" and not a.spend_test_year:     # before anything is read or written
        from .guard import Locked
        raise Locked("pass --spend-test-year: the test year is opened once, for the registered generator only")
    from . import run as R
    if a.command == "status":
        R.status()
    elif a.command == "register":
        with _record("register", vars(a)) as rec:
            path = R.register(a.series)
            rec.output(path)
        print(f"wrote {rel(path)}; commit it and testyear/ before rehearsing")
    else:
        with _record(a.command, vars(a)) as rec:
            out, path = R.rehearse() if a.command == "rehearse" else R.test(a.spend_test_year)
            rec.output(path)
            rec.result(verdict=out["verdict"])
        R.report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
