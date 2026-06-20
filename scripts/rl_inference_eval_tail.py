import csv, sys, math
def pctl(v,q):
    if not v: return float('nan')
    v=sorted(v); k=(len(v)-1)*q; f=math.floor(k); c=math.ceil(k)
    return v[int(k)] if f==c else v[f]*(c-k)+v[c]*(k-f)
import glob, os
base="/scratch/alpine/kamil@colostate.edu/vidur-rl/evals/postfix_inference_eval_eps005_rerun_20260328_182411"
print(f"{'variant':22s} {'n':>6} {'med_e2e':>8} {'p99_e2e':>9} {'p99_ttft':>9} {'p99_tpot':>9}")
for d in sorted(glob.glob(base+"/*/")):
    var=os.path.basename(d.rstrip('/'))
    rm=glob.glob(d+"*/request_metrics.csv")
    if not rm: continue
    rows=list(csv.DictReader(open(rm[0])))
    e2e=[float(r['request_e2e_time']) for r in rows]
    ttft=[float(r['prefill_e2e_time']) for r in rows]
    tpot=[(float(r['request_e2e_time'])-float(r['prefill_e2e_time']))/float(r['request_num_decode_tokens'])
          for r in rows if float(r['request_num_decode_tokens'])>0]
    print(f"{var:22s} {len(rows):>6} {pctl(e2e,.5):>8.2f} {pctl(e2e,.99):>9.1f} {pctl(ttft,.99):>9.1f} {pctl(tpot,.99):>9.3f}")
