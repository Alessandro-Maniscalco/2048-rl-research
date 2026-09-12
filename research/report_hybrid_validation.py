"""Live display for frozen100-game evaluation, with no partial average."""
import json
from pathlib import Path


def render():
    out=Path('runs/research/endgame_tablebase/hybrid_validation100')
    if not (out/'protocol.json').exists():return
    example = ''
    audit_path = out/'seed8975052/high_tile_audit.json'
    if audit_path.exists():
        a = json.loads(audit_path.read_text())
        def board(cells):
            return '<div class="board">'+''.join(
                f'<div class="tile">{value:,}</div>' if value else '<div class="tile"></div>'
                for row in cells for value in row)+'</div>'
        example = f'''<h2>Inspecting the 1,252,648-point game</h2>
        <p>Move {a['first_65536_move']:,}: <strong>right</strong> merges two 32,768 tiles.
        It earns 65,536 + 32 + 4 = <strong>65,572 raw points</strong>, then the normal random tile spawns.
        The actual game preserves the 65,536 tile exactly.</p>
        <div class="boards"><div>Before the move{board(a['before'])}</div>
        <div>After the move and spawn{board(a['after_spawn'])}</div></div>
        <p>The player survives {a['subsequent_moves']:,} more moves and earns {a['subsequent_points']:,} more points.
        All 6,996 subsequent table recommendations were re-queried with the exact large tile and matched their saved actions and probabilities;
        the remaining 3,726 positions correctly fell through to compact search. That search uses a capped high-tile representation.</p>
        <p>At an earlier position, the recorded left move had an eight-move expected return of 84.54 versus 101.05 for right.
        This is a finite-horizon disagreement, not proof of a better full-game choice. We will test a selective tactical check separately.</p>
        <p><a href="seed8975052/replay.html">Watch the complete game</a> ·
        <a href="seed8975052/high_tile_audit.json">Large-tile and tactical calculations</a> ·
        <a href="seed8975052/audit.json">Full-game seed and transition audit</a></p>'''
    comparison = ''
    comparison_path = out/'historical_comparison.json'
    if comparison_path.exists():
        c = json.loads(comparison_path.read_text())
        old, new = c['arms']
        low, high = c['independent_bootstrap95']
        interpretation = ('This does not establish a difference in average score.' if low <= 0 <= high
                          else 'The interval excludes zero, but these different implementations and seed sets do not isolate one change.')
        comparison = f'''<h2>Completed comparison with the historical reference</h2>
        <p>Historical depth-8 mean: <strong>{old['mean_score']:,.2f}</strong>.
        Hybrid mean: <strong>{new['mean_score']:,.2f}</strong>.
        Hybrid minus historical: {c['hybrid_minus_historical_mean']:+,.2f}; independent bootstrap 95% interval
        [{low:,.0f}, {high:,.0f}]. {interpretation}</p>
        <p>Each evaluation contains 100 naturally completed games. The seed sets and RNG implementations differ;
        this is not a paired comparison. Runtime setups also differ.</p>
        <img src="historical_comparison.png" alt="Final-score distributions and tile-reaching rates for two frozen players" style="width:100%;height:auto">
        <p><a href="historical_comparison.json">Exact comparison data and limitations</a></p>'''
    page = '''<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>Frozen hybrid: 100 fresh games</title>
    <style>body{font:17px system-ui;line-height:1.5;color:#23313e;max-width:1100px;margin:36px auto;padding:0 20px}
    h1{line-height:1.15}.note{background:#edf4f8;padding:18px;border-radius:10px}a{color:#245ab5}
    table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
    .table{overflow-x:auto}.boards{display:flex;flex-wrap:wrap;gap:24px}.board{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:5px;width:260px;background:#bbada0;padding:7px;border-radius:6px;margin-top:8px}.tile{aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#eee4da;font-size:15px;font-weight:700;border-radius:3px}</style></head><body><h1>Frozen hybrid: 100 fresh games</h1>
    <p>The combined player uses the old solved formation tables, then the compact search engine when no cached recommendation qualifies.
    We selected it for this larger evaluation because the eight-pair test showed much lower runtime, with an uncertain score difference.</p>
    <div class="note"><strong id="progress">Starting the evaluation.</strong><p id="mean">The mean is withheld until all 100 games finish naturally.</p>
    <p id="best"></p></div>
    <p>The policy is frozen: no new reward, neural update, tactical override or timing reset. Sixteen independent CPU workers;
    each game has a two-hour limit. Every completed game is checked against its exact seeded trajectory and raw merge scores.
    Failed or time-limited attempts remain visible.</p>
    __HISTORICAL_COMPARISON__
    __AUDITED_EXAMPLE__
    <h2>Running games</h2><div class="table"><table><thead><tr><th>Seed</th><th>Score so far</th><th>Largest tile</th><th>Moves</th></tr></thead><tbody id="active"></tbody></table></div>
    <h2>Finished attempts</h2><div class="table"><table><thead><tr><th>Seed</th><th>Result</th><th>Score</th><th>Largest tile</th><th>Seconds</th><th>Inspect</th></tr></thead><tbody id="games"></tbody></table></div>
    <p><a href="protocol.json">Frozen protocol and seeds</a> · <a href="summary.json">Summary data</a> ·
    <a href="../hybrid_comparison/index.html">Completed eight-pair comparison</a> ·
    <a href="../index.html">Research notes and move audits</a></p>
    <script>const num=x=>x==null?'—':Math.round(x).toLocaleString();
    async function data(path){const r=await fetch(path,{cache:'no-store'});return r.ok?await r.json():null;}
    async function refresh(){try{
      const [s,status,games]=await Promise.all([data('summary.json'),data('status.json'),data('games.json')]);
      if(s){document.getElementById('progress').textContent=s.completed_games+' of 100 games completed; '+s.incomplete_attempts+' incomplete attempts.';
        document.getElementById('mean').textContent=s.complete?'Mean raw score: '+num(s.mean_score)+'; 95% bootstrap interval ['+s.mean_score_bootstrap95.map(num).join(', ')+'].':s.validation_error||'The mean is withheld until all 100 games finish naturally.';
        const b=s.best_completed_game;
        if(b)document.getElementById('best').innerHTML='Best completed game so far: <a href="seed'+b.seed+'/replay.html">'+num(b.score)+' points</a>. This is a selected individual result, not the mean.';
      }
      if(status){const rows=await Promise.all((status.active||[]).map(async seed=>{const g=await data('seed'+seed+'/status.json');return '<tr><td>'+seed+'</td><td>'+num(g?.score)+'</td><td>'+num(g?.max_tile)+'</td><td>'+num(g?.moves||g?.length)+'</td></tr>';}));document.getElementById('active').innerHTML=rows.join('');}
      if(games)document.getElementById('games').innerHTML=games.slice().sort((a,b)=>a.seed-b.seed).map(g=>'<tr><td>'+g.seed+'</td><td>'+(g.complete?'Complete':'Incomplete')+'</td><td>'+num(g.score)+'</td><td>'+num(g.max_tile)+'</td><td>'+num(g.elapsed_seconds)+'</td><td><a href="seed'+g.seed+'/protocol.json">Protocol</a>'+(g.complete?' · <a href="seed'+g.seed+'/replay.html">Replay</a> · <a href="seed'+g.seed+'/audit.json">Audit</a>':'')+'</td></tr>').join('');
    }catch(e){}}refresh();setInterval(refresh,10000);</script></body></html>'''
    (out/'index.html').write_text(page.replace('__AUDITED_EXAMPLE__',example).replace('__HISTORICAL_COMPARISON__',comparison))


if __name__=='__main__':render()
