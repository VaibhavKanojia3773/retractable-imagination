"""Expanded, seeded evaluation: held-out layouts, source revocation and API timing.

No sensor retraining or method selection uses these evaluation episodes.
"""
from __future__ import annotations
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
if __package__:
    from .engine import Evidence, State, features, plan, replay, sigmoid
else:
    from engine import Evidence, State, features, plan, replay, sigmoid

ROOT=Path(__file__).resolve().parent


def wilson(successes,n):
    z=1.95996398454;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-radius),min(1.,center+radius)]


def generate_case(rng,weights,kind,case_id):
    width=int(rng.integers(24,35));height=int(rng.integers(17,24))
    wallx=int(rng.integers(11,width-8));upper=int(rng.integers(3,7));lower=int(rng.integers(height-7,height-3))
    truth=np.zeros(width*height,dtype=int)
    for y in range(height):
        for x in range(width):
            if x in (0,width-1) or y in (0,height-1) or (x==wallx and y not in (upper,lower)):
                truth[y*width+x]=1
    gate=upper*width+wallx
    blocked=kind in ('false_free','source_bundle')
    truth[gate]=int(blocked)
    # Structural map is supplied; passage occupancy is uncertain. This is not SLAM.
    prior=np.where(truth,math.log(.995/.005),math.log(.025/.975)).astype(float)
    prior[gate]=rng.uniform(.8,1.7) if blocked else math.log(.025/.975)
    source_cells=[gate]
    if kind=='source_bundle':
        for y in range(upper+1,min(upper+4,lower)):
            c=y*width+wallx
            source_cells.append(c);prior[c]=rng.uniform(1.,2.)
    events=[]
    allowed=np.array([c for c in range(len(truth)) if c not in source_cells])
    total=int(rng.integers(180,700));bad_at=int(total*rng.uniform(.08,.55))
    for i in range(total):
        if i==bad_at:
            for j,c in enumerate(source_cells):
                raw=rng.uniform(-2.8,-1.8) if blocked else rng.uniform(1.8,2.8)
                delta=float(np.clip(features(raw,.96)@weights,-8,8))
                events.append(Evidence(f'source-{case_id}-{j}',c,delta,'faulty-camera',i))
        c=int(rng.choice(allowed));q=float(rng.uniform(.4,1))
        raw=(2*truth[c]-1)*1.15+rng.normal(0,.35+1.2*(1-q))
        events.append(Evidence(f'e-{case_id}-{i}',c,float(np.clip(features(raw,q)@weights,-8,8)),'camera-A',i))
    revoked=[] if kind=='no_corruption' else [e.id for e in events if e.source=='faulty-camera']
    if kind=='no_corruption':
        # No injected wrong evidence; audit is explicitly empty.
        events=[e for e in events if e.source!='faulty-camera']
    start=upper*width+2;goal=upper*width+width-3
    before=replay(prior,events)
    prefix_plan=plan(before,width,height,start,goal)
    prefix=[start]
    for c in prefix_plan[1:6]:
        if truth[c]:break
        prefix.append(c)
    current=prefix[-1]
    return dict(width=width,height=height,truth=truth,prior=prior,events=events,revoked=revoked,
                current=current,goal=goal,before=before,prefix=prefix,kind=kind)


def score_route(path,truth):
    success=bool(path) and not any(truth[c] for c in path)
    return int(success),len(path)-1 if path else None


def task_study(weights,per_family=120,seed=20261102):
    rng=np.random.default_rng(seed);rows=[];families={};raw=[]
    names=('stale','state_only_stale_forecast','full_replay','checkpoint_replay','explicit_subtract','indexed_retract')
    for kind in ('false_free','false_occupied','source_bundle','no_corruption'):
        samples=[]
        for i in range(per_family):
            case=generate_case(rng,weights,kind,f'{kind}-{i}')
            p0=plan(case['before'],case['width'],case['height'],case['current'],case['goal'])
            target=replay(case['prior'],case['events'],case['revoked'])
            state=State(case['prior'])
            for e in case['events']:state.add(e)
            # Cache before revision. A separate baseline intentionally retains it.
            if p0:state.forecast('old-route',p0)
            invalidated=state.retract(case['revoked'])
            corrected=plan(state.logits,case['width'],case['height'],case['current'],case['goal'])
            explicit=case['before'].copy()
            ledger={e.id:e for e in case['events']}
            for key in case['revoked']:
                e=ledger[key];explicit[e.cell]-=e.delta
            j=min([idx for idx,e in enumerate(case['events']) if e.id in case['revoked']],default=len(case['events']))
            boundary=j//64*64
            prefix_state=replay(case['prior'],case['events'][:boundary])
            cp=replay(prefix_state,case['events'][boundary:],case['revoked'])
            paths={'stale':p0,'state_only_stale_forecast':p0,'indexed_retract':corrected,
                   'full_replay':plan(target,case['width'],case['height'],case['current'],case['goal']),
                   'checkpoint_replay':plan(cp,case['width'],case['height'],case['current'],case['goal']),
                   'explicit_subtract':plan(explicit,case['width'],case['height'],case['current'],case['goal'])}
            result={m:score_route(path,case['truth']) for m,path in paths.items()}
            changed=set(ledger[key].cell for key in case['revoked'])
            keep=[c for c in range(len(target)) if c not in changed]
            samples.append({'success':{m:v[0] for m,v in result.items()},'steps':{m:v[1] for m,v in result.items()},
                            'state_error':float(np.max(abs(state.logits-target))),
                            'retention_drift':float(np.max(abs(state.logits[keep]-case['before'][keep]))) if keep else 0,
                            'revoked_packets':len(case['revoked']),'prefix_actions':len(case['prefix'])-1,
                            'cached_route_invalidated':bool(invalidated),'layout':[case['width'],case['height']],
                            'changed_plan':p0!=corrected})
        summary={'n':per_family,'methods':{},'max_state_error':max(x['state_error'] for x in samples),
                 'max_retention_drift':max(x['retention_drift'] for x in samples),
                 'changed_plans':sum(x['changed_plan'] for x in samples),
                 'revoked_packets_range':[min(x['revoked_packets'] for x in samples),max(x['revoked_packets'] for x in samples)]}
        for method in names:
            successes=sum(x['success'][method] for x in samples)
            successful_steps=[x['steps'][method] for x in samples if x['success'][method]]
            summary['methods'][method]={'successes':successes,'rate':successes/per_family,'wilson95':wilson(successes,per_family),
                                        'median_successful_steps':float(np.median(successful_steps)) if successful_steps else None}
        families[kind]=summary
        raw.extend(dict(family=kind,**x) for x in samples)
    return {'seed':seed,'per_family':per_family,'total':len(raw),'families':families,'episodes':raw,
            'protocol':'Four predeclared samplers; randomized room size, partition and passage coordinates, gate priors, background observations and sensor strength; map structure supplied; initial five safe actions retained; supplied audit IDs.',
            'limits':'Controlled passage-task distribution, not general navigation prevalence. Methods share frozen learned sensor and planner. No automatic fault discovery; packet groups are revoked together, not modeled as probabilistically independent sources.'}


def api_timings(seed=20261103):
    rng=np.random.default_rng(seed);rows=[]
    for length in (256,4096,32768):
        medians={m:[] for m in ('full_replay','checkpoint_replay','explicit_subtract','indexed_retract')}
        for repeat in range(24):
            prior=np.full(504,-3.6)
            events=[Evidence(str(i),int(rng.integers(0,504)),float(rng.normal(0,.04))) for i in range(length)]
            complete=replay(prior,events);j=int(.1*length);ids={str(j)}
            boundary=j//256*256;checkpoint=replay(prior,events[:boundary])
            routes=[tuple(range(a,a+12)) for a in range(0,400,20)]
            def downstream(logits):
                # End-to-end backend scope includes route predictions AND global plan.
                probabilities=sigmoid(logits)
                values=[float(np.exp(np.log1p(-probabilities[list(p)]).sum())) for p in routes]
                path=plan(logits,28,18,4*28+3,4*28+24)
                return values,path
            methods=list(medians);rng.shuffle(methods)
            for method in methods:
                state=None
                if method=='indexed_retract':
                    state=State(prior)
                    # Setup excluded: each method starts with the same encoded history.
                    state.logits=complete.copy();state.ledger={e.id:e for e in events};state.active=set(state.ledger)
                t=time.perf_counter_ns()
                if method=='full_replay':result=replay(prior,events,ids)
                elif method=='checkpoint_replay':result=replay(checkpoint,events[boundary:],ids)
                elif method=='explicit_subtract':
                    result=complete.copy();result[events[j].cell]-=events[j].delta
                else:
                    state.retract(ids);result=state.logits
                downstream(result)
                medians[method].append((time.perf_counter_ns()-t)/1e6)
        rows.append({'length':length,'methods':{m:{'median_ms':float(np.median(v)),
                     'q25_ms':float(np.percentile(v,25)),'q75_ms':float(np.percentile(v,75)),
                     'samples_ms':v} for m,v in medians.items()}})
    return {'seed':seed,'trials_per_condition':24,'rows':rows,
            'scope':'Backend revision plus 20 route forecasts plus global A*; excludes sensing, encoding, initial history ingestion, browser rendering and network. Method order randomized. Median/IQR across 24 independent histories per length; all candidate forecasts recomputed for every method to equalize downstream work.'}


def main():
    saved=(ROOT/'web'/'demo-data.js').read_text(encoding='utf8')
    weights=np.asarray(json.loads(saved.split('=',1)[1].strip().rstrip(';'))['sensor_weights'])
    start=time.perf_counter()
    result={'version':'2.0-study','python':platform.python_version(),'numpy':np.__version__,
            'processor':platform.processor(),'platform':platform.platform(),
            'tasks':task_study(weights),'api_timing':api_timings()}
    result['wall_seconds']=time.perf_counter()-start
    (ROOT/'results'/'study.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    summary={'tasks':{k:v for k,v in result['tasks'].items() if k!='episodes'},
             'api_timing':{'rows':[{'length':r['length'],'methods':{m:{k:v for k,v in a.items() if k!='samples_ms'} for m,a in r['methods'].items()}} for r in result['api_timing']['rows']]}}
    (ROOT/'web'/'study-data.js').write_text('window.STUDY_DATA = '+json.dumps(summary)+';\n',encoding='utf8')
    print(json.dumps({'seconds':result['wall_seconds'],'episodes':result['tasks']['total'],
                      'families':{k:{m:x['successes'] for m,x in v['methods'].items()} for k,v in result['tasks']['families'].items()},
                      'max_error':max(v['max_state_error'] for v in result['tasks']['families'].values())},indent=2))

if __name__=='__main__':main()
