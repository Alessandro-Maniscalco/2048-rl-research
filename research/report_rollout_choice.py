"""Keep discovery choices separate from fresh conditional policy evaluation."""
from pathlib import Path


def render():
    out=Path('runs/research/endgame_tablebase/rollout_choice_study')
    if not out.exists():return
    (out/'index.html').write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Testing rollout-selected moves</title>
    <style>body{font:17px system-ui;line-height:1.55;max-width:1120px;margin:36px auto;padding:0 22px;color:#20313e}h1{line-height:1.2}
    .note{padding:18px;background:#fff0ca;border-radius:10px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:15px}
    th,td{padding:11px 8px;border-bottom:1px solid #ddd;text-align:left}a{color:#245ab5}</style></head><body>
    <h1>Can simulated returns choose better moves on other boards?</h1>
    <p class="note">Six recorded positions from six other games. Offline conditional evaluation, excluded from full-game scores and records.
    A move is selected on one set of simulations and evaluated on new random streams.</p>
    <p>The earlier two-board analysis showed that an eight-move window can miss a large delayed payoff.
    This follow-up tests whether choosing by longer returns helps beyond those examples, and whether it beats our strong recorded player.</p>
    <h2>How positions were selected</h2>
    <p>For each other completed game with older tables enabled, take its first recorded decision with a tile of at least 16,384,
    at least two legal actions, and chosen immediate death risk above the smallest available risk. Six games qualify; all six are included.
    The previous record game's two positions are excluded. Higher immediate risk is a selection criterion, not a claim that the move is wrong.</p>
    <h2>Two separate phases</h2>
    <p><strong>Discovery:</strong> simulate 16 fresh streams per legal first action, followed by the existing one-million-work hybrid.
    Choose the action with the largest mean reward over 8 moves and separately over 512 moves. Exact ties keep the recorded move.
    Both choices are frozen before validation starts.</p>
    <p><strong>Validation:</strong> use 128 disjoint fresh streams per legal action. Compare the frozen short-horizon choice,
    long-horizon choice and recorded move using the same 512-move raw returns. No retuning from validation results.</p>
    <p id="status" class="note">Reading live status.</p>
    <div class="scroll"><table><thead><tr><th>Position</th><th>Recorded move</th><th>8-move choice</th><th>512-move choice</th></tr></thead><tbody id="choices"></tbody></table></div>
    <h2>Independent validation</h2><p id="result">No comparison until every validation sample finishes and passes its seeded audit.</p>
    <div class="scroll"><table><thead><tr><th>Position</th><th>Recorded mean points</th><th>Short-choice mean</th><th>Long-choice mean</th><th>Long minus recorded</th></tr></thead><tbody id="values"></tbody></table></div>
    <p id="conclusion"></p>
    <p>All values are additional raw points within 512 moves under the base policy, with zero rewards after termination.
    They are not optimal Q values or complete-game returns. Six starting boards are six clusters; the many future streams
    do not turn them into hundreds of independent boards. Per-board bootstrap intervals are exploratory.</p>
    <p>Sixteen CPU workers, up to 15 minutes per phase and 120 seconds / 3 GiB per rollout. Every time-limited or failed attempt remains explicit;
    it prevents a completed aggregate. No online player is changed automatically.</p>
    <p><a href="protocol.json">Frozen positions and protocol</a> · <a href="status.json">Live status</a> ·
    <a href="choices.json">Choices frozen before validation</a> · <a href="selection_noise.json">Post-hoc sampling-noise diagnosis</a> · <a href="policy_recheck.json">Recomputed policy decisions</a> · <a href="../long_rollout_validation/index.html">Earlier independent two-board finding</a> ·
    <a href="../index.html">All research</a></p>
    <script>const names=['up','right','down','left'],num=x=>x==null?'—':Math.round(x).toLocaleString();
    async function data(p){let r=await fetch(p,{cache:'no-store'});return r.ok?await r.json():null;}
    async function refresh(){try{let [st,ch,s]=await Promise.all([data('status.json'),data('choices.json'),data('summary.json')]);
    if(st)document.getElementById('status').textContent=st.phase+(st.planned!=null?': '+st.finished+' / '+st.planned+' samples finished.':'');
    document.getElementById('choices').innerHTML=(ch?.choices||[]).map(c=>'<tr><td>'+c.case_id+'</td><td>'+names[c.recorded_action]+'</td><td>'+names[c.short_action]+'</td><td>'+names[c.long_action]+'</td></tr>').join('');
    if(s?.complete){let d=s.mean_difference_across_fixed_boards;document.getElementById('result').textContent='Across these six fixed boards: long minus recorded '+num(d.long_minus_recorded)+' points; short minus recorded '+num(d.short_minus_recorded)+'; long minus short '+num(d.long_minus_short)+'. Conditional means only; inspect each board.';
      document.getElementById('values').innerHTML=s.cases.map(c=>'<tr><td>'+c.case_id+' (move '+num(c.source_move)+')</td><td>'+num(c.mean_points.recorded)+'</td><td>'+num(c.mean_points.short)+'</td><td>'+num(c.mean_points.long)+'</td><td>'+num(c.differences.long_minus_recorded.mean)+'<br>['+c.differences.long_minus_recorded.paired_bootstrap95.map(num).join(', ')+']</td></tr>').join('');
      document.getElementById('conclusion').textContent='Decision: keep the existing player. Long-rollout selection changed two of six moves: one lost 14,117 points on fresh validation, while the other gained 556 with an interval including no gain. The failed discovery choice had an apparent 227-point edge, but paired sample SD 29,828. Sixteen simulations were too noisy to justify that override. A confidence-based selector would need a new prospective test; it has not been validated here.';}
    else if(s&&!s.complete)document.getElementById('result').textContent='Incomplete or failed audit: no aggregate comparison published.';
    }catch(e){}}refresh();setInterval(refresh,10000);</script></body></html>''')


if __name__=='__main__':render()
