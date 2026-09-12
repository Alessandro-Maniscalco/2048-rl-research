"""Expose the formation-table ablation and its causal verification gate."""
from pathlib import Path


def render():
    out=Path('runs/research/endgame_tablebase/work_table_ablation')
    if not out.exists():return
    (out/'index.html').write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Do the older tables help?</title>
    <style>body{font:17px system-ui;line-height:1.55;max-width:1100px;margin:36px auto;padding:0 22px;color:#20313e}
    h1{line-height:1.2}.note{padding:18px;background:#fff0ca;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:15px}
    th,td{padding:10px 8px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow-x:auto}a{color:#245ab5}</style></head><body>
    <h1>Do the older formation tables improve the player?</h1>
    <p class="note">One change: enable or bypass the older solved 10/11-cell formation tables.
    Both players keep the compact engine's embedded tables, heuristic and search implementation.</p>
    <p>The earlier eight-pair comparison found a runtime advantage from the older tables, but no established score benefit.
    Its adaptive search also depended on timing. This experiment uses a repeatable search-work controller in both arms.</p>
    <h2>What stays the same</h2>
    <p>Each search starts at depth three and completes successive layers until the last layer reaches one million native work counts,
    or the engine's configured maximum depth. The final layer can exceed the target. This is the same rule whenever search runs,
    not equal total runtime or equal total work: a table recommendation can avoid a search altogether.</p>
    <p>Standard 4×4 games start with two tiles, use raw merge points and exact seeded 90/10 spawns.
    There is no undo, altered starting board, new reward, neural update or new heuristic.</p>
    <h2>First verify the comparison</h2><p id="verification" class="note">Checking duplicated trajectories.</p>
    <p>Two fresh verification seeds × both players × two repeats, capped at 4,096 moves: eight diagnostic runs.
    Repeated runs must match in every board, action, executed depth and layer work count. The two policies must also behave identically
    before the first older-table intervention. Clock durations are excluded; executed search work is checked.</p>
    <p><a href="verification/protocol.json">Verification protocol</a> · <a href="verification/status.json">Verification status</a></p>
    <h2>Then compare complete games</h2><p id="progress" class="note">Full games are gated on successful verification.</p>
    <p>Sixteen fresh paired seeds, 32 games, with 16 CPU workers. Each attempt has a one-hour wall limit and a 100,000-move limit;
    unfinished attempts stay visible and prevent a complete-score comparison. No policy changes from partial results.</p>
    <p id="difference">No means before both sides finish and pass their audits.</p>
    <p id="conclusion"></p>
    <div class="scroll"><table><thead><tr><th>Player</th><th>Mean raw score</th><th>Mean seconds</th><th>Reached 32,768</th><th>Reached 65,536</th></tr></thead><tbody id="arms"></tbody></table></div>
    <div class="scroll"><table><thead><tr><th>Seed</th><th>Older tables</th><th>Score</th><th>Moves</th><th>Status</th><th>Inspect</th></tr></thead><tbody id="games"></tbody></table></div>
    <p id="links"></p>
    <p>All completed games receive exact seeded transition, reward and spawn audits. Paired prefixes are checked again after the screen.
    A sixteen-pair screen guides the next decision; it does not replace the frozen 100-game benchmarks or establish a universal ranking.</p>
    <p><a href="../late_game_comparison/index.html">Completed late-game comparison</a> · <a href="../cached_full_rank_probe/index.html">Exact-cache compute improvement</a> ·
    <a href="../index.html">All endgame research</a></p>
    <script>const num=x=>x==null?'—':Math.round(x).toLocaleString();
    async function data(p){const r=await fetch(p,{cache:'no-store'});return r.ok?await r.json():null;}
    async function refresh(){try{const [v,vs,s,status,games]=await Promise.all([data('verification/summary.json'),data('verification/status.json'),data('screen/summary.json'),data('screen/status.json'),data('screen/games.json')]);
    document.getElementById('verification').textContent=v?(v.complete?'Passed: all four duplicated pairs and both pre-intervention prefixes match.':'Verification failed; full-game comparison blocked.'):(vs?'Verification: '+vs.finished+' of 8 attempts finished.':'Verification is starting.');
    if(status){document.getElementById('progress').textContent=status.finished+' of 32 full-game attempts finished. '+status.phase;
      document.getElementById('links').innerHTML='<a href="screen/protocol.json">Frozen full-game protocol</a> · <a href="screen/status.json">Live status</a>'+(s?' · <a href="screen/summary.json">Complete comparison details</a>':'');}
    if(s?.complete){let d=s.paired_difference;document.getElementById('difference').textContent='Without older tables minus with tables: '+num(d.no_old_tables_minus_tables)+' mean points; paired bootstrap 95% interval ['+num(d.bootstrap95[0])+', '+num(d.bootstrap95[1])+']. '+d.wins+' wins, '+d.ties+' ties out of 16.';
      document.getElementById('conclusion').innerHTML='Completed: the older-table arm had a higher sample mean and lower elapsed cost, but the score interval includes zero. Keep it as the working base; this is not a new100-game benchmark. <a href="screen/seed8982012_tables1/replay.html">Watch the1,357,916-point local best</a> · <a href="screen/seed8982012_tables1/high_tile_audit.json">Detailed audit</a> · <a href="../long_rollout_validation/index.html">What longer simulated continuations reveal about its decisions</a>';
      document.getElementById('arms').innerHTML=Object.entries(s.arms).map(([n,a])=>'<tr><td>'+n.replaceAll('_',' ')+'</td><td>'+num(a.mean_score)+'</td><td>'+num(a.mean_seconds)+'</td><td>'+a.reaching_32768+'/16</td><td>'+a.reaching_65536+'/16</td></tr>').join('');}
    else if(s&&!s.complete){document.getElementById('difference').textContent='Incomplete or unverified experiment: no score means published.';}
    let rows=[];for(const g of games||[]){let path='screen/'+g.job.name;rows.push('<tr><td>'+g.job.seed+'</td><td>'+(g.job.use_tables?'Enabled':'Bypassed')+'</td><td>'+num(g.score)+'</td><td>'+num(g.length)+'</td><td>'+(g.complete?'Natural end':'Incomplete')+'</td><td><a href="'+path+'/protocol.json">Protocol</a>'+(g.exact_seeded_replay_audited?' · <a href="'+path+'/replay.html">Replay</a>':'')+(g.complete?' · <a href="'+path+'/audit.json">Audit</a>':'')+'</td></tr>');}
    for(const j of status?.active||[]){let x=await data('screen/'+j.name+'/status.json');rows.push('<tr><td>'+j.seed+'</td><td>'+(j.use_tables?'Enabled':'Bypassed')+'</td><td>'+num(x?.score)+'</td><td>'+num(x?.moves)+'</td><td>Running</td><td>—</td></tr>');}
    document.getElementById('games').innerHTML=rows.join('');}catch(e){}}refresh();setInterval(refresh,10000);</script></body></html>''')


if __name__=='__main__':render()
