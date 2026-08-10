#!/usr/bin/env python3
"""Generate Titan Inference SFT v2: 51,500 reasoning examples in Titan chat JSONL."""
import json, random, math, hashlib, zipfile
from pathlib import Path
from collections import Counter, defaultdict

SEED=20260810
random.seed(SEED)
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'data'/'sft'/'titan_inference_v2'
SYSTEM=('You are Titan, a language model built from scratch by your user. '
        'Reason carefully from supplied evidence, distinguish fact from inference, '
        'and give the most justified answer.')
rows=[]; seq=0

def add(cat,diff,domain,user,answer,rationale,tags):
    global seq
    seq+=1
    rows.append({'id':f'TITAN-INF2-{seq:07d}','category':cat,'difficulty':diff,'domain':domain,
                 'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user},
                             {'role':'assistant','content':answer}],
                 'rationale':rationale,'tags':tags})

names=['Ava','Ben','Cara','Dylan','Eli','Farah','Gina','Hugo','Iris','Jonah','Kai','Lina']
services=['auth-service','billing-api','media-worker','search-api','job-runner','checkout-api','render-worker','notification-service']
langs=['Python','TypeScript','Go','Rust','Java','C#']
dbs=['PostgreSQL','MySQL','Redis','MongoDB']
platforms=['Render','AWS','Azure','GCP','Docker','Kubernetes']

# 1. Formal deduction — 4,000
triples=[('sparrow','bird','animal'),('salmon','fish','animal'),('oak','tree','plant'),('sedan','car','vehicle'),('ruby','gem','mineral'),('laptop','computer','machine'),('copper','metal','element'),('router','network device','hardware')]
for i in range(4000):
    a,b,c=random.choice(triples); n=random.choice(names); m=i%5; cid=f'L-{i+1:04d}'
    if m==0: u=f'Case {cid}. Every {a} is a {b}. Every {b} is a {c}. {n} has a {a}. What follows necessarily?'; ans=f'The {a} is a {c}.'; rat=f'{a} → {b} and {b} → {c}, so {a} → {c}.'; d='easy'
    elif m==1: u=f'Case {cid}. All {a}s are {b}s. {n} has a {b}. Can we conclude it is a {a}?'; ans='No. The premise does not establish the converse.'; rat=f'The rule is {a} → {b}, not {b} → {a}.'; d='medium'
    elif m==2: u=f'Case {cid}. No {a} is a {c}. {n} has a {a}. Can it also be a {c}?'; ans='No.'; rat='The two classes are explicitly disjoint.'; d='easy'
    elif m==3: u=f'Case {cid}. Some {b}s are {a}s. Every {a} is a {c}. What follows about at least some {b}s?'; ans=f'At least some {b}s are {c}s.'; rat='The existential members inherit the implication.'; d='medium'
    else: u=f'Case {cid}. Every {a} is a {b}; every {b} is a {c}; every {c} is an object. {n} has a {a}. Classify it.'; ans=f'It is a {b}, a {c}, and an object.'; rat='Apply each implication in sequence.'; d='hard'
    add('deductive_logic',d,'general',u,ans,rat,['logic','deduction'])

# 2. Multi-step arithmetic — 3,500
for i in range(3500):
    rate=random.randint(8,80); hrs=random.randint(2,12); reject=random.randint(0,30); recover=random.randint(0,20); total=rate*hrs-reject+recover
    add('multi_step_reasoning','medium','general',f'Case M-{i+1:04d}. A process completes {rate} units/hour for {hrs} hours. {reject} fail validation and {recover} validated units are recovered from an earlier batch. How many validated units remain?',str(total),f'{rate}×{hrs}={rate*hrs}; -{reject}; +{recover}; result {total}.',['arithmetic','multi_step'])

# 3. Causal reasoning — 3,000
causal=[('API latency rose sharply','database connection-pool saturation','latency spikes only when the pool is exhausted; CPU and network remain normal'),('battery drain doubled','continuous background GPS','power draw returns to baseline when GPS is disabled and rises when re-enabled'),('checkout conversion dropped','payment failures','traffic quality and page speed stayed stable while payment errors rose'),('builds started failing','dependency update regression','failures began after the update and disappear with the prior lockfile'),('image jobs slowed','worker concurrency reduction','queue depth rose after concurrency was cut while per-job compute stayed constant')]
alts=['DNS','CPU saturation','network congestion','user error','random coincidence']
for i in range(3000):
    effect,cause,evidence=random.choice(causal); alt=random.choice(alts)
    add('causal_reasoning','hard','technical',f'Case C-{i+1:04d}. Outcome: {effect}. Evidence: {evidence}. Alternative: {alt}. Which explanation is better supported?',f'{cause.capitalize()} is better supported.','It matches timing and mechanism and is directly supported; the alternative lacks comparable evidence.',['causality','evidence'])

# 4. Abduction — 2,500
abduct=[('A service returns 401 after credential rotation; network tests pass and the old credential was revoked.','The service is probably still using an old or incorrect credential.'),('A page is blank after a frontend release; the API is healthy and the browser reports an uncaught import error.','The new frontend bundle likely contains a client-side import or runtime failure.'),('A nightly job stopped after its service account expired; the same job succeeds under another valid account.','The expired service account is the best explanation.'),('Timeouts occur only when the database pool reaches maximum; CPU stays below 30%.','Database connection exhaustion is the most likely bottleneck.'),('Audio drift appears only in exports longer than 40 minutes; source media is synchronized.','The export pipeline likely has a long-duration timestamp or sample-rate defect.')]
for i in range(2500):
    ev,ans=random.choice(abduct); add('abductive_reasoning','medium','mixed',f'Case A-{i+1:04d}. Evidence: {ev} What is the best current explanation? State it as a hypothesis, not certainty.',ans,'Choose the explanation that accounts for the observations with the fewest unsupported assumptions.',['abduction','best_explanation'])

# 5. Counterfactuals — 2,500
for i in range(2500):
    workers=random.randint(2,12); tasks=random.randint(workers,workers*8); mins=random.randint(3,40); seqt=tasks*mins; waves=math.ceil(tasks/workers); par=waves*mins
    add('counterfactual_reasoning','medium','technical',f'Case CF-{i+1:04d}. {tasks} equal tasks each take {mins} minutes. If {workers} workers run perfectly in parallel with no overhead, what is completion time and time saved?',f'{par} minutes, saving {seqt-par} minutes.',f'Sequential={seqt}; parallel={waves} waves×{mins}={par}.',['counterfactual','parallelism'])

# 6. Bayesian/base-rate reasoning — 2,500
for i in range(2500):
    pop=random.choice([1000,2000,5000,10000]); prev=random.choice([1,2,5,10,20]); sens=random.choice([80,85,90,95,98]); spec=random.choice([90,95,97,98,99]); diseased=pop*prev/100; tp=diseased*sens/100; fp=(pop-diseased)*(100-spec)/100; ppv=tp/(tp+fp)
    add('probabilistic_reasoning','hard','general',f'Case P-{i+1:04d}. In {pop} people, {prev}% have X. A test has {sens}% sensitivity and {spec}% specificity. Approximate P(X | positive).',f'Approximately {ppv*100:.1f}%.',f'Expected TP≈{tp:.1f}, FP≈{fp:.1f}; posterior=TP/(TP+FP)≈{ppv:.4f}.',['bayes','base_rate'])

# 7. Uncertainty calibration — 2,500
unc=[('A service became slower today. The only new fact is that a deployment happened yesterday.','The deployment is plausible, but evidence is insufficient to identify it confidently as the cause.'),('Latency rose immediately after deployment; rollback restored normal latency; redeploying reproduced the slowdown.','The deployment is strongly supported as the cause.'),('Three customers report the same UI issue, but no telemetry or reproduction is available.','The reports justify investigation, but not a confident root-cause conclusion.'),('A defect appears in every tested device of one model and none of three other tested models.','The device model is a strong explanatory factor, though the mechanism still needs confirmation.')]
for i in range(2500):
    ev,ans=random.choice(unc); add('uncertainty_calibration','hard','mixed',f'Case U-{i+1:04d}. Evidence: {ev} Give the strongest justified conclusion without overstating certainty.',ans,'Confidence should track specificity, reproducibility, and discriminating power of the evidence.',['uncertainty','calibration'])

# 8. Contradiction detection — 2,000
states=[('online','offline'),('enabled','disabled'),('locked','unlocked'),('running','stopped'),('approved','rejected')]
for i in range(2000):
    obj=random.choice(['service','device','account','job','release','sensor']); x,y=random.choice(states); t=random.randint(1,24)
    add('contradiction_detection','easy','general',f'Case K-{i+1:04d}. Record A: the {obj} was {x} at hour {t}. Record B: the same {obj}, same definition and timestamp, was {y}. Are both records consistent?','No.',f'{x} and {y} are mutually exclusive under the same-time, same-definition condition.',['contradiction','consistency'])

# 9. Evidence synthesis — 3,000
for i in range(3000):
    svc=random.choice(services); rel=random.randint(1,99); before=random.randint(80,140); after=before+random.randint(60,250); e1=random.uniform(.1,1); e2=random.uniform(5,25)
    u=f'Case E-{i+1:04d}. {svc}: latency {before}ms before R{rel} and {after}ms immediately after; error rate {e1:.1f}%→{e2:.1f}%; DB/network remain normal; rollback R{rel} restores both metrics. Synthesize the evidence.'
    add('evidence_synthesis','hard','technical',u,f'Release R{rel} is strongly implicated in the {svc} regression.','Timing, multiple changed outputs, stable alternatives, and rollback reversal converge on the release.',['evidence_synthesis','diagnostics'])

# 10. Planning — 3,000
plans=[('release a web service',['run automated tests','build release artifact','deploy to staging','run staging smoke tests','deploy to production','verify production health']),('perform a database migration',['create verified backup','test migration on a representative copy','schedule the change','run migration','verify data integrity','monitor application errors']),('launch a marketing campaign',['define target audience','prepare creative','configure attribution','run small controlled test','analyze results','scale validated variants']),('release a mobile app',['freeze release candidate','run device regression tests','prepare store metadata','submit build','address review findings','publish approved build'])]
for i in range(3000):
    goal,steps=random.choice(plans); sh=steps[:]; random.shuffle(sh)
    add('planning','medium','mixed',f'Case PL-{i+1:04d}. Goal: {goal}. Put these actions in dependency-aware order: '+'; '.join(sh)+'.',' → '.join(s.capitalize() for s in steps)+'.','Validate low-risk stages before high-impact deployment steps.',['planning','dependencies'])

# 11. Constraint optimization — 2,500
for i in range(2500):
    budget=random.randint(70,180); opts=[(n,random.randint(30,210),random.randint(60,99),random.randint(1,5)) for n in 'ABCD']; feasible=[o for o in opts if o[1]<=budget]
    if not feasible: opts[0]=('A',budget,random.randint(60,99),random.randint(1,5)); feasible=[opts[0]]
    best=max(feasible,key=lambda o:(o[2],-o[3],-o[1])); desc='; '.join(f'{n}: cost {c}, quality {q}, risk {r}/5' for n,c,q,r in opts)
    add('constraint_optimization','hard','business',f'Case O-{i+1:04d}. Budget={budget}. {desc}. Choose highest-quality feasible option; break ties by lower risk then lower cost.',f'Option {best[0]}.',f'{best[0]} ranks highest among feasible options under the stated rule.',['optimization','constraints'])

# 12. Root-cause analysis — 3,500
roots=[('Requests return 401 after credential rotation; connectivity succeeds; secret store has the new key but process environment contains the old value.','The running process is using a stale credential.'),('Memory grows after repeated image uploads; heap snapshots show retained image buffers after requests finish.','A request-lifecycle memory leak is retaining image buffers.'),('A queue grows while workers are healthy; enqueue rate is 400/s and worker throughput is 250/s.','Worker throughput is below arrival rate, causing backlog growth.'),('A scheduled job fails only in production; manual execution works; production service account lacks one required DB permission.','The production service account is missing the required database permission.'),('A UI route returns 404 after deploy; component exists, but the route is absent from the router table.','The deployment omitted route registration.')]
for i in range(3500):
    ev,ans=random.choice(roots); add('root_cause_analysis','hard','technical',f'Case RC-{i+1:04d}. Diagnostic evidence: {ev} Identify the most likely root cause.',ans,'The selected cause is directly aligned with the failing path and discriminating evidence.',['root_cause','diagnostics'])

# 13. Code-debug inference — 3,500
for i in range(3500):
    lang=random.choice(langs); m=i%5; x=random.randint(2,20)
    if m==0: u=f'Case D-{i+1:04d}, {lang}. A function expects an integer but receives string "{x}" from HTTP input; arithmetic fails or concatenates. Underlying defect?'; ans='The boundary input is not parsed or validated into the expected numeric type.'; rat='A type mismatch is introduced at the input boundary.'
    elif m==1: u=f'Case D-{i+1:04d}, {lang}. An async DB call is started but its result is used immediately without awaiting completion. What explains the failure?'; ans='The asynchronous operation is consumed before completion.'; rat='The program uses an async handle rather than the resolved result.'
    elif m==2: u=f'Case D-{i+1:04d}, {lang}. A loop removes items from the same list while iterating forward and adjacent matches are skipped. Why?'; ans='Mutating the collection during forward iteration shifts elements or iterator state, causing skips.'; rat='Removal changes positions while iteration advances.'
    elif m==3: u=f'Case D-{i+1:04d}, {lang}. A cache key includes user ID but not locale; changing locale sometimes returns the previous language. Defect?'; ans='The cache key omits locale, causing response variants to collide.'; rat='All output-affecting inputs must participate in cache identity.'
    else: u=f'Case D-{i+1:04d}, {lang}. Pagination uses offset=page*limit with 1-based pages, skipping the first {x} records when limit={x}. Fix it.'; ans='Use offset=(page-1)*limit.'; rat='Page 1 must begin at offset 0.'
    add('code_debug_inference','medium','software',u,ans,rat,['debugging','code_reasoning'])

# 14. Defensive cyber diagnostics — 3,000
cyber=[('SSH authentication failures rise 500% from many external IPs; successful logins do not increase; no new persistence artifacts appear.','The evidence is most consistent with automated credential-guessing activity, not confirmed compromise.'),('A workstation contacts a new domain every 60 seconds after a suspicious attachment is opened; EDR shows a new auto-start process.','The host is likely compromised and showing beacon-like behavior; isolate and investigate it.'),('DNS queries encode long high-entropy subdomain labels from one host while normal web traffic stays low.','The pattern is suspicious for DNS-based data transfer or tunneling; contain and inspect the host.'),('A privileged account logs in from two distant countries within minutes, including an unmanaged device.','The pattern is strongly suspicious for credential compromise or session theft.'),('An internet-facing server has a new listening service after an unplanned binary change; its hash is absent from baseline.','Treat the server as potentially compromised and investigate the unauthorized service and binary.')]
for i in range(3000):
    ev,ans=random.choice(cyber); add('cyber_diagnostics','hard','cybersecurity',f'Case CY-{i+1:04d}. Security telemetry: {ev} What is the best defensive inference and immediate priority?',ans,'Calibrate confidence and prioritize containment or verification from observed indicators.',['security','defensive_diagnostics'])

# 15. Tool selection — 1,500
tools=[('Determine whether an API endpoint is currently returning 500 errors in production.','Live logs or monitoring telemetry.','Current runtime state requires runtime evidence.'),('Count exact rows in a local CSV.','A parser or data-analysis tool.','The answer is deterministic from file contents.'),('Obtain the current exchange rate.','A live financial or web data source.','Rates change over time.'),('Find why a unit test fails reproducibly.','Run the test and inspect its stack trace.','Execution evidence localizes the failure.'),('Compare two Git commits.','Version-control diff tooling.','A diff directly identifies changed files and lines.')]
for i in range(1500):
    need,ans,rat=random.choice(tools); add('tool_selection','medium','technical',f'Case T-{i+1:04d}. Task: {need} What evidence source or tool is most appropriate?',ans,rat,['tool_use','verification'])

# 16. Self-verification — 1,500
for i in range(1500):
    x=random.randint(20,500); y=random.randint(2,30); z=random.randint(1,100); actual=x*y+z; claimed=actual+random.choice([-10,-5,-1,0,1,5,10]); ans='Correct.' if claimed==actual else f'Incorrect. Correct result: {actual}.'
    add('self_verification','easy','general',f'Case V-{i+1:04d}. Verify before accepting: {x} × {y} + {z} = {claimed}.',ans,f'Recompute: {x}×{y}={x*y}; +{z}={actual}.',['verification','arithmetic'])

# 17. Business decision reasoning — 2,500
for i in range(2500):
    visits=random.randint(1000,20000); cr1=random.uniform(1,5); cr2=cr1+random.uniform(-1,2); m1=random.randint(20,100); m2=random.randint(20,100); p1=visits*cr1/100*m1; p2=visits*cr2/100*m2; best='A' if p1>p2 else 'B'
    add('business_decision_reasoning','medium','business',f'Case B-{i+1:04d}. Two offers get {visits} visits each. A converts {cr1:.2f}% at ${m1} contribution/sale. B converts {cr2:.2f}% at ${m2}/sale. Which yields more expected contribution?',f'Offer {best}. A≈${p1:,.0f}; B≈${p2:,.0f}.','Expected contribution = visits × conversion rate × contribution per sale.',['business','unit_economics'])

# 18. Film-production reasoning — 2,000
films=[('Dialogue picture is clean but HVAC noise changes between takes; ADR budget is available.','Use dialogue cleanup where feasible and ADR for lines that cannot be repaired consistently.'),('A VFX-heavy shot has motion blur, no tracking markers, and a complex camera move.','Expect harder tracking and roto; use lens/camera metadata, feature tracking, and manual cleanup.'),('Two cameras recorded the same scene at mismatched frame rates and drift appears over long takes.','Conform footage to a deliberate project frame-rate strategy and verify sync over full takes.'),('A night exterior is underexposed and noisy, and reshooting is impossible.','Use careful denoising and selective grade recovery; irretrievably clipped detail cannot be reliably recreated.'),('Production can afford one extra shooting day or heavy overtime across three days; overtime increases fatigue risk.','Prefer the extra shooting day if cast and location constraints permit, reducing fatigue and schedule compression.')]
for i in range(2000):
    ev,ans=random.choice(films); add('film_production_reasoning','hard','cinema',f'Case F-{i+1:04d}. {ev} What is the most defensible production decision?',ans,'Weigh technical recoverability, schedule, cost, and downstream production risk.',['cinema','production_decision'])

# 19. Temporal reasoning — 1,500
for i in range(1500):
    st=random.randint(0,18); dur=random.randint(1,8); gap=random.randint(0,5); bdur=random.randint(1,6); end=st+dur+gap+bdur
    add('temporal_reasoning','medium','general',f'Case TM-{i+1:04d}. Task A starts hour {st}, lasts {dur}h. B starts {gap}h after A finishes and lasts {bdur}h. When does B finish?',f'Hour {end}.',f'A ends {st+dur}; B starts {st+dur+gap}; finishes {end}.',['temporal','sequencing'])

# 20. Long-context inference — 1,500
for i in range(1500):
    svc=random.choice(services); plat=random.choice(platforms); db=random.choice(dbs); base=random.randint(80,130); bad=base+random.randint(100,400)
    u=f'Case LC-{i+1:04d}. Incident dossier for {svc} on {plat}: 09:00 latency {base}ms/error 0.4%; 09:12 release 27.4 deployed; 09:15 latency {bad}ms/error 8.1%; {db} CPU/storage/connections normal; packet loss <0.1%; 09:31 rollback; 09:35 latency returns near {base}ms/error 0.5%; release changed request serialization. What conclusion best integrates the evidence, and what should be inspected first?'
    add('long_context_inference','hard','technical',u,'Release 27.4 is strongly implicated. Inspect the request-serialization change and downstream effects first.','The regression follows the release, alternatives remain normal, rollback reverses it, and the release contains a plausible mechanism.',['long_context','evidence_synthesis','root_cause'])

assert len(rows)==51500, len(rows)
assert len({r['messages'][1]['content'] for r in rows})==51500

# Stratified 90/5/5 split.
random.seed(SEED); bycat=defaultdict(list)
for r in rows: bycat[r['category']].append(r)
train=[]; val=[]; test=[]
for items in bycat.values():
    random.shuffle(items); n=len(items); nv=int(n*.05); nt=int(n*.05); val+=items[:nv]; test+=items[nv:nv+nt]; train+=items[nv+nt:]
random.shuffle(train); random.shuffle(val); random.shuffle(test)
assert (len(train),len(val),len(test))==(46350,2575,2575)

OUT.mkdir(parents=True,exist_ok=True)
for p in OUT.glob('*'):
    if p.is_file(): p.unlink()
shard_size=math.ceil(len(train)/10)
for i in range(10):
    with (OUT/f'train-{i+1:05d}-of-00010.jsonl').open('w',encoding='utf-8') as f:
        for r in train[i*shard_size:(i+1)*shard_size]: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
for name,data in [('validation.jsonl',val),('test.jsonl',test)]:
    with (OUT/name).open('w',encoding='utf-8') as f:
        for r in data: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

manifest={'dataset':'Titan Inference SFT v2','version':'2.0.0','seed':SEED,'format':'TitanSFTDataset-compatible chat JSONL','total_examples':len(rows),'splits':{'train':len(train),'validation':len(val),'test':len(test)},'categories':dict(sorted(Counter(r['category'] for r in rows).items())),'domains':dict(sorted(Counter(r['domain'] for r in rows).items())),'difficulty':dict(sorted(Counter(r['difficulty'] for r in rows).items())),'behavioral_restrictions_added':False,'exact_duplicate_user_prompts':0}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'README.md').write_text(f'''# Titan Inference SFT v2\n\n{len(rows):,} reasoning/inference examples in TitanAI native chat JSONL.\n\nTrain: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}\n\nCoverage includes deduction, multi-step reasoning, causality, abduction, counterfactuals, Bayesian reasoning, uncertainty calibration, contradiction detection, evidence synthesis, planning, optimization, root-cause analysis, code debugging, defensive cyber diagnostics, tool selection, self-verification, business reasoning, film-production reasoning, temporal reasoning, and long-context inference.\n\nNo behavioral-policy, refusal, or capability restriction layer is added by this dataset.\n''',encoding='utf-8')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*')) if p.is_file()}
(OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
zip_path=OUT/'Titan_Inference_SFT_v2.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(OUT.glob('*')):
        if p.name!=zip_path.name: z.write(p,arcname=f'titan_inference_sft_v2/{p.name}')
print(json.dumps(manifest,indent=2))
print(f'Generated: {OUT}')
