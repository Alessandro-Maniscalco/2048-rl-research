"""Show the new external compact player separately from published large-table claims."""
import json
from pathlib import Path


def render():
    base=Path('runs/research/endgame_tablebase');rows=[];continued=[]
    paths=sorted(base.glob('pilot_seed*/protocol.json'))+sorted(base.glob('continued_seed*/protocol.json'))
    for path in paths:
        folder=path.parent;status_path=folder/'status.json'
        state=json.loads(status_path.read_text()) if status_path.exists() else {}
        result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
        links=' · '.join(f'<a href="{folder.name}/{name}">{label}</a>' for name,label in
            [('replay.html','Replay'),('protocol.json','Provenance'),('decisions.json','Actual decisions'),('audit.json','Transition audit')]
            if (folder/name).exists())
        target=continued if folder.name.startswith('continued_') else rows
        target.append(f'<tr data-run="{folder.name}"><td>{folder.name}</td><td>{state.get("phase","starting")}</td>'
            f'<td>{state.get("score",0):,}</td><td>{state.get("moves",state.get("length",0)):,}</td>'
            f'<td>{state.get("max_tile",0):,}</td><td>{links}</td></tr>')
    headings='<tr><th>Attempt</th><th>Status</th><th>Score so far / final</th><th>Moves</th><th>Largest tile</th><th>Inspect</th></tr>'
    continuation=''
    if (base/'continuation_protocol/protocol.json').exists():
        continuation=f'''<h2>Finishing time-limited games</h2>
        <p>Every original game that reached the 30-minute time limit is resumed, regardless of its score.
        The exact board, score and spawn RNG are restored by replaying the saved legal actions from the original seed.
        <strong>The planner's adaptive timing state is reset</strong> because it was not saved in the original run.
        These are separately disclosed continued games; they do not replace the original screen results or establish its mean.
        Each gets up to 60 additional minutes. Total elapsed cost includes the original segment.</p>
        <p><a href="continuation_protocol/protocol.json">Continuation protocol</a> ·
        <a href="continuation_protocol/status.json">Live continuation status</a></p>
        <table>{headings}{''.join(continued)}</table>'''
    hybrid=''
    best_path=base/'latest_verified_best.json'
    if best_path.exists():
        best=json.loads(best_path.read_text())
        hybrid+=f'<p><strong>Latest verified local best: {best["score"]:,} points.</strong> <a href="{best["folder"]}/replay.html">Watch the full game</a> · <a href="{best["folder"]}/high_tile_audit.json">High-tile audit</a>. A selected individual game, not a mean or a world-record claim.</p>'
    if (base/'hybrid_comparison/protocol.json').exists():
        from research.report_hybrid_endgame import render as render_hybrid
        render_hybrid()
        hybrid+='<p class="note"><strong>Earlier controlled test:</strong> <a href="hybrid_comparison/index.html">Compact engine alone versus adding the older solved formation tables</a>. Eight fresh paired seeds; full uninterrupted games.</p>'
    if (base/'hybrid_validation100/protocol.json').exists():
        from research.report_hybrid_validation import render as render_validation
        render_validation()
        state_path = base/'hybrid_validation100/status.json'
        complete = state_path.exists() and json.loads(state_path.read_text()).get('phase') == 'complete'
        label = 'Completed evaluation' if complete else 'Now evaluating'
        hybrid+=f'<p><strong>{label}:</strong> <a href="hybrid_validation100/index.html">The frozen combined player on 100 fresh games</a>. No policy changes from partial results.</p>'
    if (base/'tactical_comparison/protocol.json').exists():
        from research.report_tactical_hybrid import render as render_tactical
        render_tactical()
        hybrid+='<p><strong>Separate controlled test:</strong> <a href="tactical_comparison/index.html">Does an exact eight-move tactical check improve the hybrid?</a> Twelve fresh pairs; no other policy change.</p>'
    if (base/'work_budget_repeatability/protocol.json').exists():
        from research.report_work_budget import render as render_work
        render_work()
        hybrid+='<p><strong>Next research step:</strong> <a href="work_budget/index.html">Repeatable search work and a one-million versus four-million work comparison</a>.</p>'
    if (base/'late_game_comparison/protocol.json').exists():
        from research.report_late_game import render as render_late
        render_late()
        hybrid+='<p><strong>Separate late-game analysis:</strong> <a href="late_game_comparison/index.html">Full-rank search from recorded 65,536 positions</a>. Conditional continuations, excluded from full-game rankings.</p>'
    if (base/'cached_full_rank_probe/summary.json').exists():
        from research.report_cached_full_rank import render as render_cached
        render_cached()
        hybrid+='<p><strong>Compute improvement:</strong> <a href="cached_full_rank_probe/index.html">An exact large-tile cache: unchanged answers, less repeated search</a>. Saved-query benchmark, not a game-score claim.</p>'
    if (base/'work_table_ablation').exists():
        from research.report_work_table_ablation import render as render_ablation
        render_ablation()
        hybrid+='<p><strong>Full-game comparison:</strong> <a href="work_table_ablation/index.html">Do the older formation tables help under repeatable search work?</a> Verified trajectories and sixteen fresh pairs.</p>'
    if (base/'long_rollout_validation/horizon_analysis.json').exists():
        from research.report_long_rollout import render as render_rollouts
        render_rollouts()
        hybrid+='<p><strong>New decision analysis:</strong> <a href="long_rollout_validation/index.html">Why short-horizon advice can miss large delayed rewards</a>. Same-policy rollouts, independently checked with128fresh streams per action.</p>'
    if (base/'rollout_choice_study').exists():
        from research.report_rollout_choice import render as render_choices
        render_choices()
        hybrid+='<p><strong>Completed diagnostic:</strong> <a href="rollout_choice_study/index.html">Choose moves from simulated returns, then test on independent futures</a>. Six other boards; no demonstrated improvement over the existing player.</p>'
    if (base/'work_validation100/protocol.json').exists():
        from research.report_work_validation import render as render_work_validation
        render_work_validation()
        hybrid+='<p><strong>Fresh frozen-policy evaluation:</strong> <a href="work_validation100/index.html">One-million-work hybrid on100 new games</a>. Complete results and replay audits are required before reporting a mean.</p>'
    risk_lesson=''
    diagnostic=base/'pilot_seed8972000/exact_final_move.json'
    if diagnostic.exists():
        d=json.loads(diagnostic.read_text())
        eight=next((r for r in d['results'] if r['horizon']==8),None)
        if eight:
            right=eight['actions']['right']['expected_additional_raw_score']
            left=eight['actions']['left']['expected_additional_raw_score']
            risk_lesson=f'''<h2>A correction: immediate risk is not the whole objective</h2>
            <p>The 630,020-point pilot's final right move had a 90% immediate death risk, versus 10% for left.
            We initially called this a weakness. Exact enumeration of the following eight moves changes that interpretation:</p>
            <table><tr><th>First move</th><th>Immediate death risk</th><th>Expected additional points within 8 moves</th></tr>
            <tr><td>Right (the actual choice)</td><td>90%</td><td>{right:.2f}</td></tr>
            <tr><td>Left</td><td>10%</td><td>{left:.2f}</td></tr></table>
            <p>Right's surviving branch can open a much better position. Each number optimizes future actions for raw points,
            enumerating all legal moves and random spawns, with zero value after the eight-move horizon.
            This supports right over that finite horizon; it does not prove full-game optimality.
            An immediate-risk flag is a diagnostic, not automatically a mistake.
            <a href="pilot_seed8972000/exact_final_move.json">Exact calculations for horizons 1–8</a>.</p>'''
    late_depth=''
    if (base/'late_depth_native_probe.json').exists():
        late_depth='''<h2>Investigating a late-game code path</h2>
        <p>When the sum of tile values is between 65,380 and 65,500, the upstream Python controller assigns a special deeper-search configuration.
        An independent <code>if / elif / else</code> immediately overwrites it. An isolated <code>if → elif</code> correction restored the special settings on 37 recorded search positions.</p>
        <p>This is a confirmed control-flow issue, not a confirmed score improvement. Three native board probes kept the same actions in both variants.
        Two corrected queries searched deeper; one hit its time limit and fell back to a shallower search while taking longer.
        The running 100-game evaluation uses the original frozen controller.</p>
        <p><a href="late_depth_diagnostic.json">Control-flow diagnostic</a> ·
        <a href="late_depth_native_probe.json">Actual native query measurements</a></p>'''
    page=f'''<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="60">
    <title>Compact endgame engine research</title><style>body{{font:18px system-ui;line-height:1.5;max-width:1100px;margin:40px auto;padding:0 20px;color:#202b38}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#245ab5}}</style>
    <h1>A stronger compact search player?</h1><p>The game-difficulty/2048EndgameTablebase project reports a 772,353 mean with large tables,
    and an 80% reaching rate for the 32,768 tile with its compact standalone player. These are author-reported results, not reproduced project results.
    We built and are testing its compact CoreAILogic with embedded compressed tables on this Mac.
    <a href="https://github.com/game-difficulty/2048EndgameTablebase">Primary source</a>.</p>
    {hybrid}
    <h2>Original 30-minute attempts</h2><table>{headings}{''.join(rows)}</table>
    <p>Seed 8972000 is the compatibility/runtime pilot. Seeds 8972001–8972008 form a separate prespecified eight-game screen.
    An unfinished game's score is not a completed result or an average. <a href="screen8/protocol.json">Screen protocol</a> ·
    <a href="screen8/summary.json">Live screen summary</a>.</p>
    {continuation}
    {risk_lesson}
    {late_depth}
    <h2>Why test this</h2><p>It combines adaptive look-ahead, phase-dependent board heuristics and compressed endgame probabilities.
    It permits some useful disordered formations and allocates more work to difficult positions. This changes the planning recipe
    substantially compared with merely increasing the previous player's search depth.</p>
    <h2>What actually runs</h2><p>Our exact Python game supplies the current board. Upstream CoreAILogic probes its embedded tables,
    verifies candidates and otherwise uses adaptive search. The chosen direction is applied by our unchanged Game2048;
    random spawns and raw merge rewards are produced by our environment. No undo, artificial start, reward shaping or future RNG access.</p>
    <p>The planner uses a 4-bit approximation for high tiles; our game preserves 65,536 and larger tiles exactly. Two 32,768 tiles may be
    downgraded in the search input to suggest their merge, but actual game tiles and points never change except through legal moves.
    Each decision saves its search mode, depth, duration and action. Timing-based search may vary with machine load.</p>
    <p>Portable Apple Clang C++17 build; one search thread per game process. Three local code adjustments support serial compilation,
    correct aligned allocation/deallocation, and guard missing root moves. Large table generation is not enabled.
    Five tests passed, covering 400 direction/movement checks, board/RNG isolation, high-tile scoring, terminal search,
    and exact restoration of the environment RNG for continued games.</p>
    <p><a href="../tablebase_search/index.html">Established 611,854 mean reference</a> ·
    <a href="../scaled_transformer/native_policy.html">Neural imitation results</a> ·
    <a href="../scaled_transformer/index.html">All experiments</a></p>
    <script>async function refresh(){{for(const row of document.querySelectorAll('[data-run]')){{try{{
    const r=await fetch(row.dataset.run+'/status.json',{{cache:'no-store'}});if(!r.ok)continue;const s=await r.json();
    row.children[1].textContent=s.phase;row.children[2].textContent=(s.score||0).toLocaleString();
    row.children[3].textContent=(s.moves||s.length||0).toLocaleString();row.children[4].textContent=(s.max_tile||0).toLocaleString();
    }}catch(e){{}}}}}}refresh();setInterval(refresh,5000);</script></html>'''
    (base/'index.html').write_text(page)


if __name__=='__main__':render()
