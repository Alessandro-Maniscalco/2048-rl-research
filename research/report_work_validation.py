"""Report the frozen repeatable-work evaluation without provisional means."""
from pathlib import Path


def render():
    out = Path('runs/research/endgame_tablebase/work_validation100')
    if not out.exists():
        return
    (out/'index.html').write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>100 fresh games: repeatable search work</title>
    <style>body{font:17px system-ui;line-height:1.55;max-width:1120px;margin:36px auto;padding:0 22px;color:#20313e}
    h1{line-height:1.2}.note{padding:18px;background:#fff0ca;border-radius:10px}.scroll{overflow-x:auto}
    table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px 8px;border-bottom:1px solid #ddd;text-align:left}a{color:#245ab5}</style></head><body>
    <h1>100 fresh games with repeatable search work</h1>
    <p class="note">A frozen player, with a new seed set and no tuning from these results. The 16-game screening mean of 715,240
    is promising but uncertain. This larger evaluation estimates the player's actual score distribution.</p>
    <h2>Which player?</h2><p>Use the older solved formation tables when available; otherwise use the compact engine's embedded tables
    and search. Each iterative search starts at depth 3 and finishes layers until the last layer reaches one million native work counts
    or the configured maximum depth. The last layer can overshoot. This avoids choosing search depth from wall-clock timing.</p>
    <p>The completed table comparison found a 2.68× runtime advantage, with an uncertain score difference. The earlier four-million
    work comparison cost 3.25× as much without an established score gain. We therefore froze the economical one-million version.</p>
    <p>Seeds 9020000–9020099; 16 CPU workers. Each game has up to one hour and 100,000 moves; the batch has a two-hour wall budget.
    Every failed, unfinished or unstarted attempt remains in the record and prevents a complete mean. No selected restarts or extensions.</p>
    <h2>Progress</h2><p id="status" class="note">Reading live progress.</p>
    <p id="summary">No average until all 100 games terminate naturally and pass the final source, cache and replay checks.</p>
    <p id="best"></p><p id="tiles"></p>
    <div class="scroll"><table><thead><tr><th>Seed</th><th>Raw score</th><th>Moves</th><th>Largest tile</th><th>Status</th><th>Inspect</th></tr></thead><tbody id="games"></tbody></table></div>
    <h2>What counts as evidence?</h2><p>Each replay is reproduced from its exact seed, checking every move, spawn and merge reward.
    Sampled decisions include high-tile transitions, all recorded avoidable immediate risks, and the final moves. Policy source,
    binaries and cached tables are fingerprinted before and after the batch.</p>
    <p>This is a standard full-game evaluation, with no undo or altered starting board. No neural training or new rollout intervention
    is used. The old 611,854 and 636,750 averages remain separate historical benchmarks; their different seeds do not form a paired comparison.</p>
    <p><a href="protocol.json">Frozen protocol</a> · <a href="summary.json">Summary data</a> · <a href="games.json">Every attempt</a> ·
    <a href="../work_table_ablation/index.html">Selection experiment</a> · <a href="../rollout_choice_study/index.html">Why a rollout override was rejected</a> ·
    <a href="../index.html">All research</a></p>
    <script>const num=x=>x==null?'—':Math.round(x).toLocaleString();
    async function data(p){const r=await fetch(p,{cache:'no-store'});return r.ok?await r.json():null;}
    async function refresh(){try{const [st,s,g]=await Promise.all([data('status.json'),data('summary.json'),data('games.json')]);
    if(st)document.getElementById('status').textContent=st.phase+': '+st.finished+' / '+st.planned+' attempts finished; '+(st.active||[]).length+' active.';
    if(s?.complete){document.getElementById('summary').textContent='All 100 games verified. Mean raw score '+num(s.mean_score)+'; bootstrap 95% interval ['+s.mean_score_bootstrap95.map(num).join(', ')+']; score SD '+num(s.score_std)+'. Mean length '+num(s.mean_length)+' moves. Total '+num(s.total_transitions)+' transitions.';
    document.getElementById('tiles').textContent='Tile-reaching rates: '+Object.entries(s.tile_reaching_rates).map(([t,p])=>num(+t)+': '+Math.round(p*100)+'%').join(' · ');}
    else if(st?.phase==='incomplete')document.getElementById('summary').textContent='Incomplete evaluation: every attempt is retained, but no population mean is published.';
    if(s?.best_completed_game){let b=s.best_completed_game;document.getElementById('best').innerHTML='Best verified individual game so far: <a href="seed'+b.seed+'/replay.html">'+num(b.score)+' points</a>. An individual outcome, not an average or world record.';}
    document.getElementById('games').innerHTML=(g||[]).slice().sort((a,b)=>a.seed-b.seed).map(r=>'<tr><td>'+r.seed+'</td><td>'+num(r.score)+'</td><td>'+num(r.length)+'</td><td>'+num(r.max_tile)+'</td><td>'+(r.saved_audit_checked?'Verified complete':r.stop_reason||r.error||'Incomplete')+'</td><td>'+(r.exact_seeded_replay_audited?'<a href="seed'+r.seed+'/replay.html">Replay</a> · <a href="seed'+r.seed+'/audit.json">Audit</a>':'—')+'</td></tr>').join('');
    }catch(e){}}refresh();setInterval(refresh,10000);</script></body></html>''')


if __name__ == '__main__':
    render()
