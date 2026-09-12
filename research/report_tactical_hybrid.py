"""Display the separate tactical intervention experiment and both endpoints."""
import json
from pathlib import Path


def render():
    out = Path('runs/research/endgame_tablebase/tactical_comparison')
    if not (out/'protocol.json').exists(): return
    diagnostic = ''
    path = out/'transition_diagnostic.json'
    if path.exists():
        d = json.loads(path.read_text())
        diagnostic = f'''<h2>What the transition audit found</h2>
        <p><strong>All {d['pairs_diverging_before_any_override']} pairs chose different moves before any tactical override.</strong>
        Their initial boards matched, but the wall-clock controller completed different search depths.
        The observed score difference cannot cleanly isolate the tactical intervention.</p>
        <p>We independently recomputed all {d['overrides_recomputed']} overrides. The stored eight-move values and chosen actions matched.
        The tactical player's 65,536 tile was reached before its first override, so that milestone cannot be credited to the check.</p>
        <p>The experiment supplies no reason to promote the tactical rule. We are testing repeatable search-work allocation before further comparisons.
        <a href="transition_diagnostic.json">Per-pair divergence and exact action calculations</a>.</p>'''
    page = '''<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Does a tactical check help?</title>
    <style>body{font:17px system-ui;line-height:1.5;max-width:1100px;margin:36px auto;padding:0 20px;color:#23313e}
    table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}
    .note{padding:18px;background:#edf4f8;border-radius:10px}.table{overflow-x:auto}a{color:#245ab5}</style></head><body>
    <h1>Does a selective tactical check improve full games?</h1>
    <p>The control is our frozen hybrid: solved formation tables, then compact adaptive search.
    The candidate adds one check when the proposed move risks immediate death more than another legal move.</p>
    <p>It enumerates every legal move and random spawn for eight moves, maximizing expected additional raw points.
    It changes the proposal only if that complete calculation favors another action. It preserves ties and retains
    the original proposal if the calculation exceeds 500,000 states or 30 seconds. Risk triggers the check; it is not the objective.</p>
    <p><strong>Why test this?</strong> A restricted table objective can disagree with near-term points.
    But an eight-move horizon can also sacrifice a valuable long-term formation. We need complete game results to judge.</p>
    <div class="note"><p id="progress">Starting.</p><p id="comparison">No comparison until all 24 games complete and pass audits.</p></div>
    __TRANSITION_DIAGNOSTIC__
    <div class="table"><table><thead><tr><th>Policy</th><th>Mean score</th><th>Mean seconds</th><th>32,768 games</th>
    <th>65,536 games</th><th>Overrides / checks</th></tr></thead><tbody id="arms"></tbody></table></div>
    <h2>Running games</h2><div class="table"><table><thead><tr><th>Policy</th><th>Seed</th><th>Score so far</th><th>Largest tile</th></tr></thead><tbody id="active"></tbody></table></div>
    <h2>Finished attempts</h2><div class="table"><table><thead><tr><th>Policy</th><th>Seed</th><th>Status</th><th>Score</th><th>Seconds</th><th>Inspect</th></tr></thead><tbody id="games"></tbody></table></div>
    <p>Twelve fresh paired seeds, sixteen independent CPU workers, up to two hours per game. Ordinary game rules and exact seeded audits.
    No changes to table thresholds, compact search settings or rewards. No partial averages, selected restarts or automatic extension.
    Search timing depends on machine load, so matching seeds do not force identical decision prefixes.</p>
    <p><a href="protocol.json">Protocol and frozen fingerprints</a> · <a href="summary.json">Summary data</a> ·
    <a href="../hybrid_validation100/index.html">Completed frozen 100-game evaluation</a> · <a href="../index.html">Research notes</a></p>
    <script>const num=x=>x==null?'—':Math.round(x).toLocaleString();
    const name=t=>t?'tactical':'control', folder=g=>name(g.tactical)+'_seed'+g.seed;
    async function data(p){const r=await fetch(p,{cache:'no-store'});return r.ok?await r.json():null;}
    async function refresh(){try{const [s,status,games]=await Promise.all([data('summary.json'),data('status.json'),data('games.json')]);
    if(s){document.getElementById('progress').textContent=s.finished+' of 24 attempts finished; '+s.incomplete_attempts+' incomplete.';
      const d=s.paired_difference;document.getElementById('comparison').textContent=s.complete&&d?
      'Tactical minus control mean: '+num(d.mean)+' points; paired 95% interval ['+d.bootstrap95.map(num).join(', ')+']; '+d.wins+' wins and '+d.ties+' ties.':
      s.validation_error||'No comparison until all 24 games complete and pass audits.';
      document.getElementById('arms').innerHTML=s.complete?Object.entries(s.arms).map(([n,a])=>'<tr><td>'+n+'</td><td>'+num(a.mean_score)+'</td><td>'+num(a.mean_seconds)+'</td><td>'+a.reaching_32768+'/12</td><td>'+a.reaching_65536+'/12</td><td>'+a.tactical_overrides+' / '+a.tactical_checks+'</td></tr>').join(''):'';}
    if(status){const rows=await Promise.all((status.active||[]).map(async g=>{const r=await data(folder(g)+'/status.json');return '<tr><td>'+name(g.tactical)+'</td><td>'+g.seed+'</td><td>'+num(r?.score)+'</td><td>'+num(r?.max_tile)+'</td></tr>';}));document.getElementById('active').innerHTML=rows.join('');}
    if(games)document.getElementById('games').innerHTML=games.slice().sort((a,b)=>a.seed-b.seed||a.tactical-b.tactical).map(g=>'<tr><td>'+name(g.tactical)+'</td><td>'+g.seed+'</td><td>'+(g.complete?'Complete':'Incomplete')+'</td><td>'+num(g.score)+'</td><td>'+num(g.elapsed_seconds)+'</td><td><a href="'+folder(g)+'/protocol.json">Protocol</a>'+(g.complete?' · <a href="'+folder(g)+'/replay.html">Replay</a> · <a href="'+folder(g)+'/audit.json">Audit</a>':'')+'</td></tr>').join('');
    }catch(e){}}refresh();setInterval(refresh,10000);</script></body></html>'''
    (out/'index.html').write_text(page.replace('__TRANSITION_DIAGNOSTIC__',diagnostic))


if __name__ == '__main__': render()
