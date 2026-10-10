import json,sys,numpy as np
d=json.load(open(sys.argv[1])); P=d["points"]
print(d["scenario"],"H",d["H"],"points",len(P),"inflight median",np.median([p["inflight"] for p in P]),"wall/pt %.1fs"%np.mean([p["wall_s"] for p in P]), "adv %.2f drain %.2f"%(np.mean([p["adv_s"] for p in P]),np.mean([p["drain_s"] for p in P])))
print("policy==forced(policy choice) identical:", all(p["policy_matches_forced_same"] for p in P))
for m in ("mean","p90","max","sq"):
    g=np.array([p[m]["gain_vs_policy"] for p in P]); r=np.array([p[m]["policy_rank"] for p in P])
    sp=np.array([p[m]["spearman_cens_full"] for p in P]); cb=np.mean([p[m]["cens_picks_full_best"] for p in P])
    spread=np.array([1-p[m]["best"]/p[m]["worst"] for p in P])
    print(f"{m:5s} gain best-vs-policy median {np.median(g):.3f} mean {g.mean():.3f} (>1%: {np.mean(g>0.01):.2f}); policy is best {np.mean(r==0):.2f}, rank median {np.median(r)}; best-worst spread median {np.median(spread):.3f}; censored~full spearman median {np.nanmedian(sp):.2f}, censored picks best {cb:.2f}")
