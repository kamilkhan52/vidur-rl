import tempfile, time, copy, sys, os, pickle
from evolve import evaluate as ev
from evolve.spike_a11 import argv_for
from evolve.sim_fork import build_simulator, _precopy_block_pool
sc = ev.Scenario("Tool", 8, 0.75, 0, 600, 1, model="l3_8b")
with tempfile.TemporaryDirectory() as t:
    h = build_simulator(argv_for(sc, t)); h.run_to_decision(300)
    s = h.sim
    t0=time.perf_counter(); memo=h._shared_memo(); t1=time.perf_counter()
    print("shared memo incl. block precopy: %.3f s, memo entries %d" % (t1-t0, len(memo)))
    for name in ("_cluster_metric_store","_scheduler","_cluster","_event_queue","_request_generator","_config"):
        m=h._shared_memo(); t0=time.perf_counter(); copy.deepcopy(getattr(s,name), m); print(f"  deepcopy {name}: {time.perf_counter()-t0:.3f} s")
    gs=s._scheduler
    for rid,rs in list(gs._replica_schedulers.items())[:1]:
        for k,v in rs.__dict__.items():
            m=h._shared_memo(); t0=time.perf_counter(); copy.deepcopy(v,m); dt=time.perf_counter()-t0
            if dt>0.003: print(f"    replica0.{k}: {dt:.3f} s")
    ms=s._cluster_metric_store
    for k,v in ms.__dict__.items():
        m=h._shared_memo(); t0=time.perf_counter(); copy.deepcopy(v,m); dt=time.perf_counter()-t0
        if dt>0.003: print(f"    metrics.{k}: {dt:.3f} s")
    # os.fork cost
    import resource
    for i in range(3):
        t0=time.perf_counter(); pid=os.fork()
        if pid==0:
            os._exit(0)
        t1=time.perf_counter(); os.waitpid(pid,0); t2=time.perf_counter()
        print(f"os.fork: parent return {1e3*(t1-t0):.1f} ms, fork+exit+wait {1e3*(t2-t0):.1f} ms")
