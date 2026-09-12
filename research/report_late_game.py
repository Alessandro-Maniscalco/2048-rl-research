"""Clearly separate saved-position continuation results from full-game records."""
from pathlib import Path


def render():
    out=Path('runs/research/endgame_tablebase/late_game_comparison')
    if not (out/'protocol.json').exists():return
    (out/'index.html').write_text('''<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Late-game representation test</title>
    <style>body{font:17px system-ui;line-height:1.5;max-width:1100px;margin:36px auto;padding:0 20px;color:#23313e}.note{padding:18px;background:#fff0ca;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}.table{overflow-x:auto}a{color:#245ab5}</style></head><body>
    <h1>Can a full-rank planner improve play after 65,536?</h1>
    <p class="note"><strong>Saved-position continuations, not full games from two tiles.</strong>
    Every run starts at one of three recorded first-65,536 positions, with a fresh future spawn seed.
    These results are excluded from full-game averages and score records. Compare additional points earned.</p>
    <p>The compact engine represents ranks of 32,768 and above as unmergeable markers after the first 65,536 tile.
    Our real game always keeps exact tiles, but this approximation can prevent the planner from considering another large merge.</p>
    <p>The candidate uses the existing solved tables first. On a table miss, it runs the older cache-corrected search with full tile ranks,
    depth eight and its built-in orientation-adaptive heuristic. This changes both the representation and the fallback search algorithm.
    It is still heuristic, probability-pruned search; it does not calculate exact full-game Q values.</p>
    <p><strong>Representation examples:</strong> a real row [65,536, 32,768, 32,768, 0] can merge left for 65,536 points;
    [65,536, 65,536, 0, 0] can merge for 131,072 points. The capped compact representation prevents both merges.
    The new native move implementation matches the exact Python game on these examples and 320 randomized direction checks.</p>
    <p><a href="../full_rank_probe.json">Synthetic examples and recorded-board cost probes</a> ·
    <a href="../full_rank_bridge/build.json">Build provenance and fingerprints</a></p>
    <p><a href="../cached_full_rank_probe/index.html">Separate cache prototype and completed-control transition audit</a>:
    exact reuse of identical search work accelerates recorded-board queries. This comparison kept its original implementation throughout.</p>
    <h2>Controlled continuation experiment</h2><p id="progress" class="note">Starting 24 continuations: three positions × four fresh future streams × two players.</p>
    <p id="difference">No means before all continuations finish and pass their audits.</p>
    <div class="table"><table><thead><tr><th>Fallback</th><th>Mean additional points</th><th>Mean seconds</th><th>Reached 131,072</th></tr></thead><tbody id="arms"></tbody></table></div>
    <p id="cases"></p>
    <p id="conclusion"></p>
    <div class="table"><table><thead><tr><th>Position / future seed</th><th>Fallback</th><th>Additional points</th><th>Largest tile</th><th>Status</th><th>Inspect</th></tr></thead><tbody id="games"></tbody></table></div>
    <p>Both arms use the same saved position and future seed in each pair. The control uses the four-million-work compact planner.
    Every recorded move, raw reward and new tile is checked by replaying the same saved position and future RNG.
    There are only three starting positions; their four streams are not twelve independent board samples. We report per-position differences.</p>
    <p>Sixteen CPU workers; up to two hours and 50,000 continuation moves per attempt. No selected restarts or omitted failures.
    A positive result here would motivate fresh full-game testing, not immediate promotion to the main benchmark.</p>
    <p><a href="protocol.json">Protocol, starting positions and future seeds</a> · <a href="summary.json">Conditional summary</a> ·
    <a href="../work_budget/index.html">Completed work-budget comparison and new full-game best</a> · <a href="../index.html">Research notes</a></p>
    <script>const num=x=>x==null?'—':Math.round(x).toLocaleString();const label=x=>x?'Full rank, depth 8':'Compact, four million';
    const folder=g=>'case'+(g.case_id??g.case.id)+'_seed'+g.future_seed+'_'+(g.full_rank?'full':'compact');
    async function data(p){const r=await fetch(p,{cache:'no-store'});return r.ok?await r.json():null;}
    async function refresh(){try{const [s,status,games]=await Promise.all([data('summary.json'),data('status.json'),data('games.json')]);
    if(s){document.getElementById('progress').textContent=s.finished+' of 24 saved-position attempts finished. Excluded from full-game score records.';
      document.getElementById('difference').textContent=s.complete?'Full-rank minus compact mean additional points: '+num(s.paired_mean_difference)+'. Only three starting positions; inspect consistency below.':s.validation_error||'No means before all continuations finish and pass their audits.';
      document.getElementById('arms').innerHTML=s.complete?Object.entries(s.arms).map(([n,a])=>'<tr><td>'+n+'</td><td>'+num(a.mean_additional_points)+'</td><td>'+num(a.mean_seconds)+'</td><td>'+a.reached_131072+'/12</td></tr>').join(''):'';
      document.getElementById('cases').textContent=s.complete?s.per_case.map(c=>'Position '+c.case_id+': '+num(c.paired_mean_difference)+' additional-point difference').join('; '):'';}
    if(s?.complete){document.getElementById('conclusion').innerHTML='Completed: no clear benefit from replacing the late compact search. Mean additional points were lower with this full-rank heuristic, and results varied across three starting positions. Neither arm built another 32,768 tile. This experiment is closed without promotion. <a href="control_transition_audit.json">Control tail audit</a> · <a href="full_rank_transition_audit.json">Full-rank tail audit</a> · <a href="../work_table_ablation/index.html">Next: isolate the older tables throughout full games</a>';}
    const rows=[];for(const g of games||[]){rows.push('<tr><td>'+g.case_id+' / '+g.future_seed+'</td><td>'+label(g.full_rank)+'</td><td>'+num(g.additional_points)+'</td><td>'+num(g.max_tile)+'</td><td>'+(g.complete?'Natural end':'Incomplete')+'</td><td><a href="'+folder(g)+'/protocol.json">Protocol</a>'+(g.complete?' · <a href="'+folder(g)+'/replay.html">Continuation replay</a> · <a href="'+folder(g)+'/audit.json">Audit</a>':'')+'</td></tr>');}
    for(const g of status?.active||[]){const x=await data(g.name+'/status.json');rows.push('<tr><td>'+g.case.id+' / '+g.future_seed+'</td><td>'+label(g.full_rank)+'</td><td>'+num(x?.additional_points)+'</td><td>'+num(x?.max_tile)+'</td><td>Running</td><td>—</td></tr>');}document.getElementById('games').innerHTML=rows.join('');
    }catch(e){}}refresh();setInterval(refresh,10000);</script></body></html>''')


if __name__=='__main__':render()
