import re

with open("/home/kamil/vidur-rl-working/scripts/rl_vs_baseline_postfix.py", "r") as f:
    code = f.read()

# Fix the column names (the dataframe has mean_e2e, not e2e_mean after aggregation)
code = code.replace("e2e_mean", "mean_e2e")
code = code.replace("plots/EXP02_RL_vs_Baseline", "plots/EXP02_RL_vs_Baseline_Postfix")
code = code.replace("grad_", "") # Just in case

# Fix the rl parsing for postfix
# the RL trace names might have -PF
# Let's adjust rl_filtered trace names
trace_fix = """# Remove -PF suffix if it exists in the rl df
rl_filtered['trace'] = rl_filtered['trace'].str.replace('-PF', '')
rl_filtered['model'] = rl_filtered['model'].str.replace('-PF', '')
"""
code = code.replace("rl_filtered['scale_round'] = rl_filtered['scale'].round(1)", trace_fix + "rl_filtered['scale_round'] = rl_filtered['scale'].round(1)")


with open("/home/kamil/vidur-rl-working/scripts/rl_vs_baseline_postfix.py", "w") as f:
    f.write(code)
