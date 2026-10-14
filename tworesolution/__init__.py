"""Two-resolution rainfall generator (code_plan/TWO_RESOLUTION_PLAN.md).

A coarse diffusion model makes 30-minute totals 32 hours at a time; a fine model, v14 with the totals as an extra
input, fills in the 5-minute detail; each half-hour's six values are scaled to its total. Run as
`python -m tworesolution <command>` from the repository root; see README.md.
"""
