"""One command: fit a tiny sensor, run controls, export reproducible demo data."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

if __package__:
    from .engine import Evidence, State, features, fit_sensor, plan, replay, scene, sigmoid
else:
    from engine import Evidence, State, features, fit_sensor, plan, replay, scene, sigmoid

ROOT = Path(__file__).resolve().parent


def timed(fn, repeats=9):
    fn()
    values = []
    for _ in range(repeats):
        t = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns()-t)/1000)
    return float(np.median(values)), float(np.percentile(values, 90))


def benchmark(seed=711):
    rng = np.random.default_rng(seed)
    rows = []
    for length in (256, 4096, 32768):
        cells = rng.integers(0, 2048, length)
        delta = rng.normal(0, .12, length)
        prior = rng.normal(0, .1, 2048)
        events = [Evidence(str(i), int(cells[i]), float(delta[i])) for i in range(length)]
        complete = replay(prior, events)
        # Materialize every checkpoint once. Setup time/storage are reported.
        t = time.perf_counter()
        checkpoints = {0: prior.copy()}
        value = prior.copy()
        for i,e in enumerate(events):
            value[e.cell] += e.delta
            if (i+1) % 256 == 0:
                checkpoints[i+1] = value.copy()
        setup_ms = (time.perf_counter()-t)*1000
        index = State(prior)
        for e in events:
            index.add(e)
        for fraction in (.1,.5,.9):
            j = int(length*fraction)
            target = replay(prior, events, {str(j)})
            boundary = (j//256)*256
            def full():
                return replay(prior, events, {str(j)})
            def checkpoint():
                return replay(checkpoints[boundary], events[boundary:], {str(j)})
            def sufficient():
                result = complete.copy()
                result[cells[j]] -= delta[j]
                return result
            def indexed():
                e = index.ledger[str(j)]
                result = index.logits.copy()
                result[e.cell] -= e.delta
                return result
            methods = {}
            for name,fn in [("full_replay",full),("checkpoint_replay",checkpoint),
                             ("explicit_subtract",sufficient),("indexed_retract",indexed)]:
                med,p90 = timed(fn)
                methods[name] = {"median_us":med,"p90_us":p90,
                                 "max_logit_error":float(np.max(np.abs(fn()-target)))}
            rows.append({"length":length,"revoke_fraction":fraction,"methods":methods,
                         "checkpoint_setup_ms":setup_ms,
                         "checkpoint_bytes":sum(a.nbytes for a in checkpoints.values()),
                         "numeric_evidence_payload_bytes":int(cells.nbytes+delta.nbytes),
                         "state_bytes":int(complete.nbytes),
                         "memory_note":"numeric arrays only; Python ledger/dependency overhead excluded"})
    return rows


def cache_trial(seed=19):
    rng = np.random.default_rng(seed)
    prior = np.full(2048, -4.)
    routes = [tuple(range(i*20, i*20+16)) for i in range(64)]
    events = [Evidence("probe", 4, -2.)]
    state = State(prior)
    state.add(events[0])
    for i,p in enumerate(routes): state.forecast(str(i),p)
    before = {i:state.forecast(str(i),p).copy() for i,p in enumerate(routes)}
    affected = state.retract(["probe"])
    fresh = State(prior)
    error, unchanged = 0.,0.
    for i,p in enumerate(routes):
        v,ref = state.forecast(str(i),p),fresh.forecast(str(i),p)
        error = max(error, abs(v['survival']-ref['survival']))
        if str(i) not in affected:
            unchanged = max(unchanged,abs(v['survival']-before[i]['survival']))
    return {"routes":len(routes),"invalidated":len(affected),"retained":len(routes)-len(affected),
            "max_forecast_error":error,"unaffected_forecast_drift":unchanged,
            "note":"locality-controlled cache test, not a generic sparsity claim"}


def task_trials(w, n=200, seed=813):
    rng = np.random.default_rng(seed)
    base = scene(w)
    width,height = base['width'],base['height']
    output = {m:[] for m in ['stale','full_replay','checkpoint_replay','explicit_subtract','indexed_retract']}
    decisions_changed = 0
    for k in range(n):
        prior = np.asarray(base['prior']).copy()
        # Vary prior and observation strengths, not just random frame appearance.
        prior[base['bad_cell']] = rng.uniform(.8,1.7)
        events = [Evidence(e['id'],e['cell'],e['delta']) for e in base['events'][:-1]]
        bad = Evidence('bad',base['bad_cell'],float(np.clip(features(rng.uniform(-2.8,-1.8),.95)@w,-8,8)))
        events.append(bad)
        before = replay(prior,events)
        # Five actual safe actions were already executed under the bad belief.
        current = 4*width+8
        oldpath = plan(before,width,height,current,base['goal'])
        fixed = replay(prior,events,{'bad'})
        state = State(prior)
        for e in events: state.add(e)
        state.retract(['bad'])
        states = {'stale':before,'full_replay':fixed,'checkpoint_replay':replay(prior,events,{'bad'}),
                  'explicit_subtract':before-np.eye(1,len(before),base['bad_cell'])[0]*bad.delta,
                  'indexed_retract':state.logits}
        for method,belief in states.items():
            path = plan(belief,width,height,current,base['goal'])
            collision = any(base['truth'][c] for c in path)
            output[method].append(int(bool(path) and not collision))
        decisions_changed += int(oldpath != plan(fixed,width,height,current,base['goal']))
    return {"n":n,"seed":seed,"success":{m:float(np.mean(v)) for m,v in output.items()},
            "changed_plans":decisions_changed,
            "scope":"selected invalid-free-gate family; revocation ID supplied; five past actions retained",
            "checkpoint_note":"task outcomes use identical corrected inference; timing is measured separately"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=20261101)
    args=parser.parse_args()
    started=time.perf_counter()
    w,training=fit_sensor(args.seed)
    training['wall_seconds']=time.perf_counter()-started
    result={"version":"1.0","seed":args.seed,"python":platform.python_version(),
            "platform":platform.platform(),"training":training,
            "timing_scope":"state repair only, pre-encoded evidence, single process, warm median of 9 runs; not end-to-end planning latency",
            "benchmark":benchmark(),"cache":cache_trial(),"tasks":task_trials(w),
            "limits":["independent static occupancy factors", "revocation identifiers externally supplied",
                      "exact operator is not learned", "explicit subtraction is algebraically equivalent",
                      "task family is selected to expose correction, not deployment prevalence",
                      "memory figures exclude Python bookkeeping", "no real robot or large VLM evaluated"]}
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    demo={"scene":scene(w),"sensor_weights":w.tolist(),"evaluation":result}
    (ROOT/'web'/'demo-data.js').write_text('window.DEMO_DATA = '+json.dumps(demo)+';\n',encoding='utf-8')
    print(json.dumps({"training_seconds":training['wall_seconds'],"sensor_accuracy":training['test_accuracy'],
                      "cache":result['cache'],"tasks":result['tasks'],
                      "total_seconds":time.perf_counter()-started},indent=2))


if __name__=='__main__': main()
