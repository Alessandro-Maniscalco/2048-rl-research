"""Display the frozen-table/compact-planner comparison without partial means."""
import json
from pathlib import Path


def render():
    out=Path('runs/research/endgame_tablebase/hybrid_comparison')
    protocol_path=out/'protocol.json'
    if not protocol_path.exists():return
    protocol=json.loads(protocol_path.read_text());rows=[]
    diagnostic=''
    risk_path=out/'risk_horizon_diagnostic.json'
    if risk_path.exists():
        risk=json.loads(risk_path.read_text())
        if risk.get('complete'):
            diagnostic=f'''<h2>What the move audits reveal</h2>
            <p>Across the eight combined-player games, {risk['number_of_risky_decisions']} decisions had higher immediate death risk than another legal move.
            Exact eight-move enumeration finished for {risk['exact8_completed']} of those boards; the rest hit the explicit computation limit.
            In {risk['chosen_is_best8']} completed case, the risky choice maximized expected points over that horizon.
            Immediate risk therefore cannot be used as an automatic mistake label.</p>
            <p>These finite-horizon diagnostics do not establish full-game optimality. The playing policies remain unchanged.
            <a href="risk_horizon_diagnostic.json">Every flagged board and horizon calculation</a> ·
            <a href="../index.html">Worked correction of the 630k game's final move</a>.</p>'''
    recheck=out/'table_recommendation_recheck.json'
    if recheck.exists():
        d=json.loads(recheck.read_text())
        if d.get('complete'):
            diagnostic+=f'''<p>All {d['total_table_decisions_rechecked']:,} frozen-table recommendations were queried again from their actual saved boards;
            actions, probabilities and table types match. <a href="table_recommendation_recheck.json">Table consistency check</a>.</p>'''
    for seed in protocol['seeds']:
        for hybrid,label in [(False,'Compact alone'),(True,'Tables + compact')]:
            name=('hybrid' if hybrid else 'compact')+f'_seed{seed}'
            rows.append(f'<tr data-run="{name}"><td>{seed}</td><td>{label}</td>'
                        '<td>Queued</td><td>—</td><td>—</td><td>—</td><td></td></tr>')
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Can solved formations improve compact search?</title><style>
    body{font:17px system-ui;line-height:1.55;color:#23313e;max-width:1150px;margin:36px auto;padding:0 20px}
    h1{line-height:1.15}a{color:#245ab5}table{border-collapse:collapse;width:100%;font-size:15px}
    td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.table{overflow-x:auto}
    .note{padding:18px;background:#edf4f8;border-radius:10px}.diagram{font:16px ui-monospace,monospace;white-space:pre-wrap}
    </style></head><body><h1>Can solved formations improve compact search?</h1>
    <p>Both players receive the same kind of exact 4×4 board and play ordinary 2048.
    We change one decision rule: give the combined player access to the older engine's solved formation tables.</p>
    <div class="note"><strong id="summary">Waiting for full results.</strong>
    <p>A game's score while running is not a final result. We show arm averages only after every assigned game in that arm finishes naturally.
    This is an eight-pair screen, not a new 100-game benchmark.</p></div>
    <h2>What each player does</h2><div class="diagram">Compact alone:   Board → compact tables / adaptive search → move
Tables + compact: Board → old solved-table recommendation, if available → move
                       → otherwise the same compact engine → move</div>
    <p>The older tables store a move and a probability of completing a restricted formation-building goal.
    The 11-cell table qualifies at probability ≥ 0.9; otherwise the 10-cell table qualifies above zero.
    Missing entries fall back to the compact engine. We do not generate or write table entries.</p>
    <p><strong>Why it might help:</strong> solved formations can protect large tiles and organize smaller tiles over a longer horizon than shallow search.
    They also avoid expensive searches on matching boards. <strong>Why it might hurt:</strong> their restricted goals may interfere with a better plan from the compact engine.</p>
    <h2>Live games</h2><div class="table"><table><thead><tr><th>Seed</th><th>Player</th><th>Status</th><th>Score</th><th>Largest tile</th><th>Seconds</th><th>Inspect</th></tr></thead><tbody>__ROWS__</tbody></table></div>
    <h2>Fairness and limits</h2><p>Eight fresh seeds, two policies, 16 single-thread CPU workers; each game has a two-hour limit and a 100,000-move cap.
    Both use the original 90%/10% tile spawns, raw merge scores and natural game-over rules. No undo, artificial start, safety filter, neural training or planner timing reset.
    Adaptive search depends on machine load, so matching seeds do not guarantee identical decisions or search timing.
    Both complete endpoints and the paired score differences are required before drawing conclusions.</p>
    __DIAGNOSTIC__
    <p><a href="protocol.json">Frozen protocol</a> · <a href="summary.json">Live comparison data</a> ·
    <a href="../index.html">Compact-engine pilots and resumed games</a> ·
    <a href="../../tablebase_search/index.html">Established 611,854 mean reference</a></p>
    <script>
    const number=x=>x==null?'—':Math.round(x).toLocaleString();
    async function refresh(){
      for(const row of document.querySelectorAll('[data-run]')){
        try{const url=row.dataset.run+'/';const r=await fetch(url+'status.json',{cache:'no-store'});if(!r.ok)continue;
          const s=await r.json();row.children[2].textContent=s.phase;
          row.children[3].textContent=number(s.score);row.children[4].textContent=number(s.max_tile);
          row.children[5].textContent=number(s.elapsed_seconds);
          row.children[6].innerHTML='<a href="'+url+'protocol.json">Protocol</a>';
          if(s.complete!==undefined)row.children[6].innerHTML+=' · <a href="'+url+'replay.html">Replay</a>';
        }catch(e){}
      }
      try{const r=await fetch('summary.json',{cache:'no-store'});if(!r.ok)return;const s=await r.json();
        let text=s.finished+' of '+s.planned+' attempts finished. ';
        for(const [key,arm] of Object.entries(s.arms)){
          if(arm.complete)text+=(key==='compact'?'Compact alone':'Tables + compact')+' mean: '+number(arm.mean_score)+'. ';
        }
        if(s.paired_difference)text+='Paired difference: '+number(s.paired_difference.mean)+'; 95% bootstrap interval ['+s.paired_difference.bootstrap95.map(number).join(', ')+'].';
        document.getElementById('summary').textContent=text;
      }catch(e){}
    }refresh();setInterval(refresh,5000);
    </script></body></html>'''
    (out/'index.html').write_text(page.replace('__ROWS__',''.join(rows)).replace('__DIAGNOSTIC__',diagnostic))


if __name__=='__main__':render()
