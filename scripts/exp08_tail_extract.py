import csv, sys, math, re
def pctl(v,q):
    if not v: return ''
    v=sorted(v); k=(len(v)-1)*q; f=math.floor(k); c=math.ceil(k)
    return v[int(k)] if f==c else v[f]*(c-k)+v[c]*(k-f)
def mean(v): return sum(v)/len(v) if v else ''
def parse(p):
    parts=p.split('/')
    ci=next(i for i,t in enumerate(parts) if re.match(r'^\d+b-\d+R-',t))  # 7b- or 70b-
    model,repl,trace=parts[ci].split('-')
    sched=parts[ci-1]
    scale=next(t for t in parts if t.startswith('scale'))[5:]
    return sched,model,repl,trace,scale
hdr=['scheduler','model','replicas','trace','scale','run_path','n_requests',
     'mean_e2e','p50_e2e','p95_e2e','p99_e2e',
     'mean_ttft','p50_ttft','p95_ttft','p99_ttft',
     'mean_tpot','p50_tpot','p95_tpot','p99_tpot',
     'queue_delay_mean','queue_delay_p99','queue_fraction','prefill_hit_ratio','n_decode_used']
w=csv.writer(sys.stdout); w.writerow(hdr)
skipped=0
for p in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    rm=p+'/request_metrics.csv'
    try:
        rows=list(csv.DictReader(open(rm)))
        if not rows: skipped+=1; continue
        e2e=[float(r['request_e2e_time']) for r in rows]
        ttft=[float(r['prefill_e2e_time']) for r in rows]
        sd=[float(r['request_scheduling_delay']) for r in rows]
        pf=[float(r['request_num_prefill_tokens']) for r in rows]
        ca=[float(r['request_num_prefill_tokens_cached']) for r in rows]
        tpot=[(float(r['request_e2e_time'])-float(r['prefill_e2e_time']))/float(r['request_num_decode_tokens'])
              for r in rows if float(r['request_num_decode_tokens'])>0]
        sched,model,repl,trace,scale=parse(p)
        me=mean(e2e); qd=mean(sd)
        w.writerow([sched,model,repl,trace,scale,p,len(rows),
            me,pctl(e2e,.5),pctl(e2e,.95),pctl(e2e,.99),
            mean(ttft),pctl(ttft,.5),pctl(ttft,.95),pctl(ttft,.99),
            mean(tpot),pctl(tpot,.5),pctl(tpot,.95),pctl(tpot,.99),
            qd,pctl(sd,.99),qd/me if me else '',sum(ca)/sum(pf) if sum(pf) else '',len(tpot)])
    except Exception as e:
        skipped+=1
sys.stderr.write(f"skipped={skipped}\n")
