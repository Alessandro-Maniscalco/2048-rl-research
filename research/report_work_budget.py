"""Readable live report for deterministic-work research and its limitations."""
import json
from pathlib import Path


def render():
    base=Path('runs/research/endgame_tablebase')
    if not (base/'work_budget_repeatability/protocol.json').exists():return
    out=base/'work_budget';out.mkdir(exist_ok=True)
    best=''
    summary_path=base/'work_budget_comparison/summary.json'
    if summary_path.exists() and json.loads(summary_path.read_text()).get('complete'):
        records=json.loads((base/'work_budget_comparison/games.json').read_text())
        record=max(records,key=lambda r:r['score'])
        name=record['job']['name']
        best=f'''<p class="note"><strong>Highest completed game in this work-budget comparison: {record['score']:,} points.</strong>
        This is a selected individual game, not an average or a world-record claim.
        <a href="../work_budget_comparison/{name}/replay.html">Watch it play</a> ·
        <a href="../work_budget_comparison/{name}/audit.json">Exact seed and transition audit</a> ·
        <a href="../work_budget_comparison/{name}/high_tile_audit.json">65,536 merge and subsequent table checks</a>.</p>'''
    page='''<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Repeatable search work</title>
    <style>body{font:17px system-ui;line-height:1.5;max-width:1100px;margin:36px auto;padding:0 20px;color:#23313e}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}.table{overflow-x:auto}.note{padding:18px;background:#edf4f8;border-radius:10px}a{color:#245ab5}</style></head><body>
    <h1>Make search work repeatable, then test whether more work helps</h1>
    <p>Every pair in the tactical experiment diverged before its first intervention: the wall-clock search completed different depths.
    To remove that source of variation, this candidate finishes successive search layers and stops when a completed layer reaches a work target.</p>
    <p>It starts at depth three, or a smaller configured maximum, and preserves the native maximum-depth rules.
    The targets are <strong>one million</strong> and <strong>four million</strong> native nodes per final layer.
    These are <strong>soft targets</strong>: a completed layer can overshoot, and previous layers also cost work.
    Direct embedded-table verification remains. Node counts are engine work counters, not hardware instructions.</p>
    <p>The solved tables, board heuristics, native binaries, standard game rules and raw score stay fixed.
    There is no tactical override or late-depth correction. Search wall time is recorded but does not select the next depth.</p>
    <h2>First: repeatability and runtime</h2><p id="repeat" class="note">Checking two seeds, two budgets and two independent repeats, each capped at 4,096 moves. These are not full-game score benchmarks.</p>
    <p><a href="../work_budget_repeatability/protocol.json">Capped-trajectory protocol</a> · <a href="../work_budget_repeatability/games.json">Runtime and partial-game records</a> · <a href="../work_budget_late_probe.json">Recorded late-board probes</a></p>
    <h2>Then: full-game comparison</h2><p id="screen" class="note">Not yet started. Requires successful repeatability and runtime checks.</p>
    __BEST_GAME__
    <div class="table"><table><thead><tr><th>Work target</th><th>Mean score</th><th>Mean seconds</th><th>32,768 games</th><th>65,536 games</th></tr></thead><tbody id="arms"></tbody></table></div>
    <div class="table"><table><thead><tr><th>Run</th><th>Seed</th><th>Score</th><th>Largest tile</th><th>Status</th><th>Replay</th></tr></thead><tbody id="games"></tbody></table></div>
    <p>Eight fresh paired seeds compare the work targets. All games must end naturally and pass exact seeded audits before means appear.
    Repeatability is necessary for useful comparisons; it does not prove this allocator is a stronger player.</p>
    <p><a href="../index.html">Research notes</a> · <a href="../tactical_comparison/index.html">Completed tactical test</a> · <a href="../hybrid_validation100/index.html">Frozen 636,750 mean reference</a></p>
    <script>const num=x=>x==null?'—':Math.round(x).toLocaleString();async function data(p){const r=await fetch('../'+p,{cache:'no-store'});return r.ok?await r.json():null;}
    async function refresh(){try{const r=await data('work_budget_repeatability/summary.json');if(r)document.getElementById('repeat').textContent=r.complete?'All four duplicated trajectories matched in every board, action, completed depth and recorded work count. Capped at 4,096 moves; not a full-game score comparison.':'Repeatability check did not pass. Inspect the saved records.';
    const [s,status,games]=await Promise.all([data('work_budget_comparison/summary.json'),data('work_budget_comparison/status.json'),data('work_budget_comparison/games.json')]);
    if(s?.complete){const d=s.paired_difference;document.getElementById('screen').textContent='Four-million minus one-million mean: '+num(d.mean)+' points; paired 95% interval ['+d.bootstrap95.map(num).join(', ')+']. '+d.wins+' wins and '+d.ties+' ties.';
      document.getElementById('arms').innerHTML=Object.entries(s.arms).map(([n,a])=>'<tr><td>'+num(+n)+'</td><td>'+num(a.mean_score)+'</td><td>'+num(a.mean_seconds)+'</td><td>'+a.reaching_32768+'/8</td><td>'+a.reaching_65536+'/8</td></tr>').join('');
    }else if(status){document.getElementById('screen').textContent=status.finished+' of 16 attempts finished. No averages before all complete and pass audits.';document.getElementById('arms').innerHTML='';}
    const rows=[];for(const g of games||[]){rows.push('<tr><td>'+num(g.job.work_target)+'</td><td>'+g.job.seed+'</td><td>'+num(g.score)+'</td><td>'+num(g.max_tile)+'</td><td>'+(g.complete?'Complete':'Incomplete')+'</td><td><a href="../work_budget_comparison/'+g.job.name+'/replay.html">Replay</a></td></tr>');}
    for(const g of status?.active||[]){const x=await data('work_budget_comparison/'+g.name+'/status.json');rows.push('<tr><td>'+num(g.work_target)+'</td><td>'+g.seed+'</td><td>'+num(x?.score)+'</td><td>'+num(x?.max_tile)+'</td><td>Running</td><td>—</td></tr>');}document.getElementById('games').innerHTML=rows.join('');
    }catch(e){}}refresh();setInterval(refresh,10000);</script></body></html>'''
    (out/'index.html').write_text(page.replace('__BEST_GAME__',best))


if __name__=='__main__':render()
