"""v19: the v14 recipe with a one-day window (code_plan/ONE_DAY_WINDOW_PLAN.md).

One change from v14: each generated piece is 288 five-minute steps (one day) instead of 64 (5h20), made possible
by the user's 24x24 pretrained checkpoint. Run as `python -m oneday <command>` from the repository root.
"""
