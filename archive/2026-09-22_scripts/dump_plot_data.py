import pandas as pd
import re
import json

df = pd.read_csv("/home/kamil/vidur-rl-working/graduation_summary.csv")
baseline_df = pd.read_csv("/home/kamil/vidur-rl-working/plots/priority2_all.csv")
if "e2e_mean" in baseline_df.columns:
    baseline_df = baseline_df.rename(columns={"e2e_mean": "mean_e2e"})

dump_data = {}

for config_name in df["config"].unique():
    cdf = df[df["config"] == config_name].sort_values("scale", ascending=False)
    if cdf.empty: continue
    
    rl_scales = cdf["scale"].tolist()
    rl_vals = cdf["mean_e2e"].tolist()
    
    config_data = {
        "RL": {"scales": rl_scales, "values": rl_vals},
        "Best_Heuristic": {"scales": [], "values": []},
        "Random": {"scales": [], "values": []}
    }
    
    m = re.match(r"grad_(\d+b)-(\d+R)-([A-Za-z0-9]+)_", config_name)
    if m:
        b_model, b_rep, b_trace = m.group(1), m.group(2), m.group(3)
        b_data = baseline_df[(baseline_df["model"] == b_model) & 
                             (baseline_df["replicas"] == b_rep) & 
                             (baseline_df["trace"] == b_trace)]
        
        if not b_data.empty:
            for s in sorted(b_data["scale"].unique(), reverse=True):
                s_data = b_data[b_data["scale"] == s]
                if not s_data.empty:
                    config_data["Best_Heuristic"]["scales"].append(s)
                    config_data["Best_Heuristic"]["values"].append(s_data["mean_e2e"].min())
                    
                    random_data = s_data[s_data["scheduler"] == "RANDOM"]
                    if not random_data.empty:
                        config_data["Random"]["scales"].append(s)
                        config_data["Random"]["values"].append(random_data["mean_e2e"].min())
                        
    dump_data[config_name] = config_data

with open("/home/kamil/vidur-rl-working/plot_data_dump.json", "w") as f:
    json.dump(dump_data, f, indent=2)

print("Dumped data for config: grad_7b-16R-Synth_h512_512_256_128_re2e_time")
print(json.dumps(dump_data["grad_7b-16R-Synth_h512_512_256_128_re2e_time"], indent=2))
print("\nDumped data for config: grad_7b-8R-Synth_h512_512_256_128_re2e_time")
print(json.dumps(dump_data.get("grad_7b-8R-Synth_h512_512_256_128_re2e_time", {}), indent=2))
