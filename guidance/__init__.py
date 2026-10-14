"""guidance: carry storms across v14's 5h20 pieces by steering each new piece, while it is generated, to agree
with the end of the previous one (reconstruction guidance, Ho et al. 2022). No training: v14's weights never
change; only the strength w is chosen, by the pilot's written rules.

Plan: code_plan/GUIDANCE_PLAN.md. Usage: guidance/README.md.
"""
