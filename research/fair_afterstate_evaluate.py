import json,time,torch,hashlib,html
from pathlib import Path
from rl2048.afterstate_compare import NeuralValue,TupleValue,evaluate_batch
B=Path(__file__).resolve().parents[1]/'runs/research/fair_afterstate_hour'
deadline=json.loads((B/'manifest.json').read_text())['deadline_epoch'];torch.set_num_threads(2)
run=B/'paired_seed0'
if not (run/'training.json').exists():run=B/'pilot_mps'
meta=json.loads((run/'training.json').read_text());results=[]
# Fixed final pilot, no tuning against these fresh games. Both receive same seeds.
selection=dict(run=run.name,training=meta,chosen_before_test=True,hashes={})
for kind,file in [('neural','value.pt'),('table','weights.npy')]:selection['hashes'][kind]=hashlib.sha256((run/kind/file).read_bytes()).hexdigest()
(B/'selection.json').write_text(json.dumps(selection,indent=2))
models={'neural':NeuralValue.load(run/'neural','mps'),'table':TupleValue.load(run/'table')}
for depth,count in [(1,100),(2,16)]:
 for kind,model in models.items():
  if time.time()>deadline-4:break
  # Allocate equal evaluation time caps; partial runs never yield a mean score.
  cap=min(20 if depth==1 else 15,max(0,(deadline-time.time()-4)/2))
  r=evaluate_batch(model,range(8300000,8300000+count),depth,deadline=time.time()+cap)
  r.update(model=kind,training_transitions=meta['transitions'],seed_count=count)
  results.append(r);(B/'results.json').write_text(json.dumps(results,indent=2))
  print(kind,depth,r['summary'],flush=True)
lines=['# Fair afterstate comparison — bounded pilot','',f"Both fresh models received {meta['transitions']:,} identical transitions.",
 'Same afterstate TD targets (gamma1), D4 symmetries, reward units and exact planner. Optimizers and parameter counts differ; this is one seed and a short budget. No pretrained reasoning model or teacher transfer was run.','',
 '| Model | Depth | Finished games | Mean raw score | Seconds |','|---|---:|---:|---:|---:|']
for r in results:
 s=r['summary'];lines.append(f"| {r['model']} | {s['depth']} | {s['completed_episodes']}/{s['episodes']} | {s['mean_score']} | {s['seconds']:.2f} |")
lines+=['','Partial evaluations have no mean and must not be ranked. Depth2 uses16 games; depth1 uses100. Same-depth comparisons share seeds. Training and evaluation stop at the original one-hour deadline.','',
 'Learning: the representation is now tested behind the same afterstate formulation and planner, but this pilot cannot establish asymptotic superiority. Full three-seed comparisons and pretrained teacher transfer remain unperformed.']
text='\n'.join(lines);(B/'journal.md').write_text(text)
(B/'index.html').write_text('<meta charset="utf-8"><title>Fair afterstate pilot</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>')
