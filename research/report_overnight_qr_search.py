"""Render the frozen QR-DQN search pilot from completed games and live progress."""
from html import escape
import json
from pathlib import Path


def planning_row(label, result, requested):
    """Only rank complete suites; show higher-tile play and its inference cost."""
    episodes = result.get('episodes', [])
    count = sum(e['complete'] for e in episodes)
    if result.get('complete'):
        mean = f'{result["mean_score"]:,.2f}'
        rate = f'{100 * sum(e["max_tile"] >= 4096 for e in episodes) / len(episodes):.0f}%'
        minutes = f'{sum(e["seconds"] for e in episodes) / 60:.1f}'
    else:
        mean = rate = minutes = 'Pending'
    return f'<tr><td>{escape(label)}</td><td>{count}/{requested}</td><td>{mean}</td><td>{rate}</td><td>{minutes}</td></tr>'


def render():
    base = Path(__file__).resolve().parents[1] / 'runs/research/overnight_qr_search'
    if not (base / 'direct.json').exists():
        return
    direct = json.loads((base / 'direct.json').read_text())['summary']
    rows = [f'<tr><td>No planning</td><td>8/8</td><td>{direct["mean_score"]:,.2f}</td><td>—</td><td>—</td></tr>']
    for depth in (1, 2, 3):
        file = base / 'pilot' / f'depth{depth}.json'
        r = json.loads(file.read_text()) if file.exists() else {}
        rows.append(planning_row(f'{depth}-step exact planning', r, 8))
    progress_file = base / 'pilot/progress.json'
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {'status': 'Preparing pilot'}
    confirmation = ''
    if (base / 'confirmation_protocol.json').exists():
        baseline = json.loads((base.parent / 'scaled_transformer/qr_compare_qr_dqn_score_seed0/evaluation.json').read_text())['summary']['mean_score']
        confirmation = (f'<h2>Separate 100-game confirmation</h2><p>Same 100 selection seeds as the direct model, '
            f'whose mean is {baseline:,.2f}. These games differ from the eight-game pilot above. '
            'Two CPU search workers share the Mac with GPU training; timings reflect that workload.</p>'
            '<table><tr><th>Planning depth</th><th>Complete games</th><th>Mean score</th><th>Reached 4096</th><th>Total game minutes</th></tr>')
        for depth in (1, 2):
            file = base / 'confirmation' / f'depth{depth}.json'
            result = json.loads(file.read_text()) if file.exists() else {}
            confirmation += planning_row(f'{depth}-step planning', result, 100)
        confirmation += '</table>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>2048 · Overnight QR planning</title>
<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:35px auto;padding:0 24px;background:#faf8ef;color:#544c44}a{color:#197970}td,th{text-align:left;padding:12px;border-bottom:1px solid #d9d2c5}table{width:100%}pre{white-space:pre-wrap}</style>
<h1>Can planning improve the frozen QR-DQN?</h1><p><a href="../scaled_transformer/index.html">Training dashboard and research journal</a> · <a href="../scaled_transformer/td_horizon.html">TD-horizon comparison</a></p>
<p>The same frozen model, previously averaging 10,136 on 100 games, plays eight new matched screening games with zero/one/two/three steps of planning. Search enumerates legal moves and all 2/4 tile spawns, averages chance outcomes, and uses the mean predicted QR return at its leaves. No weights change. Reward remains points/128; no corner/snake bonus is added in this first comparison.</p>
<p>This is an eight-game screen, not a 100-game confirmation. Scores below are averages of complete games only when all eight finish; partial scores are not ranked. CPU planning uses two threads alongside GPU training. Benchmark times include that concurrent workload. Three-step planning is promising, but eight games cannot establish a reliable advantage across games or training seeds.</p>
<table><tr><th>Policy</th><th>Completed games</th><th>Mean raw score</th><th>Reached 4096</th><th>Total game minutes</th></tr>''' + ''.join(rows) + '''</table><p>Most recently saved pilot game progress (a partial-game snapshot, not an evaluation mean; completed results are above):</p><pre>''' + escape(json.dumps(progress, indent=2)) + '''</pre><p><a href="benchmark.json">Per-decision benchmark</a> · <a href="protocol.json">Experiment protocol</a> · <a href="pilot/config.json">Pilot configuration</a></p></html>'''
    audit_links = ''.join(f'<li><a href="audits/{p.parent.name}/index.html">{p.parent.name}: replay and move decisions</a></li>'
                         for p in sorted((base / 'audits').glob('*/audit.json')))
    audits = '<h2>Policy decision audits</h2><ul>' + audit_links + '</ul>' if audit_links else ''
    if (base / 'replay_depth1_confirmed.html').exists():
        audits = '<p><a href="replay_depth1_confirmed.html">Watch the confirmed one-step planner (saved complete game)</a></p>' + audits
    improved = ''
    followup_file = base / 'improved_leaf_followup.json'
    if followup_file.exists():
        followup = json.loads(followup_file.read_text())
        if followup.get('root'):
            directory = base / followup['root']
            direct_mean = json.loads((directory / 'direct.json').read_text())['summary']['mean_score']
            training_moves = json.loads((directory / 'frozen/metadata.json').read_text())['experiment']['training_transitions']
            original_mean = json.loads((base / 'confirmation/depth1.json').read_text())['mean_score']
            file = directory / 'evaluation/depth1.json'
            result = json.loads(file.read_text()) if file.exists() else {}
            count = sum(e['complete'] for e in result.get('episodes', []))
            mean = f'{result["mean_score"]:,.2f}' if result.get('complete') else 'Pending'
            improved = (f'<h2>Can a better learned model improve shallow planning?</h2>'
                f'<p>Same Transformer architecture and one-step search, with weights from {training_moves/1e6:.2f} million training transitions '
                'instead of 4.19 million. Both use the same 100 selection games. Extra training experience is part of this comparison.</p>'
                f'<table><tr><th>Policy</th><th>Complete games</th><th>Mean score</th></tr>'
                f'<tr><td>New model, direct action</td><td>100/100</td><td>{direct_mean:,.2f}</td></tr>'
                f'<tr><td>Original model, one-step planning</td><td>100/100</td><td>{original_mean:,.2f}</td></tr>'
                f'<tr><td>New model, one-step planning</td><td>{count}/100</td><td>{mean}</td></tr></table>')
            second_file=directory/'evaluation_depth2/depth2.json'
            if (directory/'depth2_protocol.json').exists():
                second=json.loads(second_file.read_text()) if second_file.exists() else {}
                completed=sum(e['complete'] for e in second.get('episodes',[]))
                second_mean=f'{second["mean_score"]:,.2f}' if second.get('complete') else 'Pending'
                improved=improved.replace('</table>',f'<tr><td>New model, two-step planning</td><td>{completed}/100</td><td>{second_mean}</td></tr></table>')
                improved+='<p>The new one-step policy reached 2048 in 91% of these games but reached 4096 in none. The two-step follow-up tests whether deeper search improves that higher-tile limit, as well as mean score.</p>'
    shaped = ''
    shaped_file = base / 'shaped_leaf_followup.json'
    if shaped_file.exists():
        protocol = json.loads(shaped_file.read_text())
        if protocol.get('root'):
            file = base / protocol['root'] / 'evaluation/depth1.json'
            result = json.loads(file.read_text()) if file.exists() else {}
            original = json.loads((base / 'confirmation/depth1.json').read_text())
            shaped = ('<h2>Do corner and snake rewards help the planner?</h2>'
                '<p>Both QR models learned for 4.19 million transitions with the same architecture and training seed. '
                'Both use one-step planning on the same 100 selection games. The only recipe change is the corner/snake '
                'reward, used consistently in training and planning with shaping strength 2. Scores below are actual merge points.</p>'
                '<table><tr><th>Reward recipe</th><th>Complete games</th><th>Mean score</th><th>Reached 4096</th><th>Total game minutes</th></tr>'
                + planning_row('Ordinary merge reward', original, 100)
                + planning_row('Merge reward plus corner/snake potential', result, 100)
                + '</table><p><a href="shaped_leaf_followup.json">Hypothesis, control and time budget</a></p>')
    risk = ''
    risk_root = base / 'quantile_risk_24117248'
    if (risk_root / 'results.json').exists():
        results = json.loads((risk_root / 'results.json').read_text())
        risk = ('<h2>Can a different use of the 51 predictions improve decisions?</h2>'
            '<p>The same frozen 24.12M-transition QR model chooses directly from its predictions. '
            'No training, inputs, reward or planning change. All four variants use the same 100 selection seeds. '
            'Lower-half averaging favors predicted worse outcomes; upper-half averaging favors predicted better outcomes. '
            'These are return predictions, not confidence intervals for the model. '
            'This is a cheap adaptation of <a href="https://proceedings.mlr.press/v80/dabney18a.html">risk-sensitive distributional RL</a>, '
            'not a reproduction of that paper’s training method.</p>'
            '<table><tr><th>Decision rule</th><th>Complete games</th><th>Mean raw score</th><th>Reached 2048</th><th>Replay</th></tr>')
        labels = {'mean': 'Average all 51 predictions (control)', 'lower_half': 'Average the lower half',
                  'median': 'Middle prediction (median)', 'upper_half': 'Average the upper half'}
        for mode, result in results.items():
            s = result['summary']
            mean = f'{s["mean_score"]:,.2f}' if s['complete'] else 'Pending'
            rate = f'{s["reaching_2048"]:.0%}' if s['complete'] else 'Pending'
            replay = risk_root / f'replay_{mode}.html'
            link = f'<a href="{risk_root.name}/{replay.name}">Watch</a>' if replay.exists() else 'Pending'
            risk += f'<tr><td>{labels[mode]}</td><td>{s["completed_episodes"]}/100</td><td>{mean}</td><td>{rate}</td><td>{link}</td></tr>'
        risk += '</table><p>Replays use one separate diagnostic seed. A single replay may favor a different variant than the 100-game mean.</p>'
    approximate = ''
    approx_root = base / 'approximate_depth34'
    if (approx_root / 'protocol.json').exists():
        approximate = ('<h2>Can selective deeper planning improve score at a manageable cost?</h2>'
            '<p>The original frozen model plays the same eight pilot seeds. First test a probability cutoff of 0.01 '
            'at three steps against the completed exact three-step result. Then increase the maximum depth to four '
            'with the cutoff unchanged. Unlikely branches use the Q estimate earlier; their probability mass is retained. '
            'These are approximate, up-to-depth policies. Four steps are not expanded on every branch.</p>'
            '<table><tr><th>Policy</th><th>Complete games</th><th>Mean raw score</th><th>Reached 4096</th><th>Total game minutes</th></tr>')
        original = json.loads((base / 'pilot/depth3.json').read_text())
        approximate += planning_row('Exact three-step control', original, 8)
        for depth in (3, 4):
            file = approx_root / 'evaluation' / f'depth{depth}.json'
            result = json.loads(file.read_text()) if file.exists() else {}
            approximate += planning_row(f'Up to {depth} steps, probability cutoff 0.01', result, 8)
        approximate += ('</table><p><a href="approximate_depth34/protocol.json">Hypothesis and budget</a> · '
            '<a href="search_cost_screen/benchmark.json">Fixed-state cost benchmark</a></p>')
    gamma = ''
    gamma_file = base / 'gamma1_leaf_followup.json'
    if gamma_file.exists():
        protocol = json.loads(gamma_file.read_text())
        if protocol.get('root'):
            file = base / protocol['root'] / 'evaluation/depth1.json'
            result = json.loads(file.read_text()) if file.exists() else {}
            control = json.loads((base / 'confirmation/depth1.json').read_text())
            gamma = ('<h2>Does an undiscounted value model improve one-step planning?</h2>'
                '<p>Both models use the same architecture, training seed and 4.19M training transitions. '
                'Both plan one step on the same 100 selection games. Change gamma consistently during training '
                'and planning from 0.99 to 1: later merge points retain their full weight in the objective. '
                'The direct policies scored 10,136 and 11,142 respectively; that does not establish their search ranking.</p>'
                '<table><tr><th>Discount recipe</th><th>Complete games</th><th>Mean raw score</th><th>Reached 4096</th><th>Total game minutes</th></tr>'
                + planning_row('Gamma 0.99 control', control, 100)
                + planning_row('Gamma 1 throughout', result, 100)
                + '</table><p><a href="gamma1_leaf_followup.json">Hypothesis and budget</a></p>')
    parallel = ''
    parallel_root = base / 'gamma1_depth2_parallel'
    if (parallel_root / 'protocol.json').exists():
        from research.aggregate_search_shards import aggregate
        result = aggregate(parallel_root)
        control = json.loads((base / 'confirmation/depth2.json').read_text())
        parallel = ('<h2>Can deeper planning improve the undiscounted model’s consistency?</h2>'
            '<p>At one-step depth, gamma 1 reached 4096 in 8% of games versus 3% for gamma 0.99, '
            'but reached 2048 less often and had a similar mean score. This follow-up tests whether two-step planning '
            'improves that balance. Four independent workers each play 25 disjoint games with one CPU thread; '
            'the same 100 selection seeds are covered exactly once.</p>'
            '<table><tr><th>Two-step policy</th><th>Complete games</th><th>Mean raw score</th><th>Reached 4096</th><th>Worker game minutes</th></tr>'
            + planning_row('Gamma 0.99 control', control, 100)
            + planning_row('Gamma 1, four parallel game shards', result, 100) + '</table>')
        if result['complete']:
            parallel += f'<p>Parallel evaluation wall time: {result["wall_seconds"]/60:.1f} minutes. Worker game minutes sum concurrent work and are not elapsed wall time.</p>'
        parallel += '<p><a href="gamma1_depth2_parallel/protocol.json">Hypothesis, resource allocation and worker metadata</a></p>'
    deeper_confirmation = ''
    root = base / 'approximate_depth4_parallel'
    if (root/'protocol.json').exists():
        from research.aggregate_search_shards import aggregate
        result = aggregate(root)
        deeper_confirmation = ('<h2>Confirming selective four-step planning on 100 games</h2>'
            '<p>The complete eight-game pilot averaged 63,185.5 versus 45,438 for selective three-step search. '
            'This larger evaluation uses the common 100 selection seeds and the same original frozen model. '
            'The 0.01 cutoff keeps each branch’s probability but uses the neural value early on unlikely branches. '
            'Four workers run resumable two-hour review chunks; the estimated full evaluation can take several hours. '
            'Partial games never receive a mean score.</p>'
            '<table><tr><th>Policy</th><th>Complete games</th><th>Mean raw score</th><th>Reached 4096</th><th>Worker game minutes</th></tr>'
            + planning_row('Original model, exact two-step', json.loads((base/'confirmation/depth2.json').read_text()), 100)
            + planning_row('Original model, selective up-to-four-step', result, 100) + '</table>'
            '<p><a href="approximate_depth4_parallel/protocol.json">Reason, budget and workers</a></p>')
        comparison=root/'completed_comparison.json'
        if comparison.exists():
            deeper_confirmation+='<p>'+escape(json.loads(comparison.read_text())['learning'])+'</p>'
    new_leaf = ''
    file = base/'low_exploration_leaf_38273024/evaluation/depth1.json'
    if file.exists():
        result = json.loads(file.read_text())
        new_leaf = ('<h2>Does the lower-exploration policy also improve with planning?</h2>'
            '<p>The new direct policy averaged 25,294 points. This independent evaluation adds exact one-step '
            'planning to its frozen weights, using the same 100 games as the older leaf models.</p>'
            '<table><tr><th>One-step policy</th><th>Complete games</th><th>Mean raw score</th><th>Reached 4096</th><th>Game minutes</th></tr>'
            + planning_row('Original 4M model', json.loads((base/'confirmation/depth1.json').read_text()), 100)
            + planning_row('Earlier 24.12M model', json.loads((base/'improved_leaf_24117248/evaluation/depth1.json').read_text()), 100)
            + planning_row('Low-exploration 38.27M model', result, 100) + '</table>'
            '<p><a href="low_exploration_leaf_followup.json">Protocol and follow-up</a></p>')
    gamma_pilot = ''
    root = base/'gamma1_approximate_depth23_pilot'
    if (root/'protocol.json').exists():
        gamma_pilot = ('<h2>Can selective deeper search extend the gamma 1 result?</h2>'
            '<p>After the 52,644-point exact two-step result, this pilot compares maximum depths two and three '
            'with the same gamma 1 model and 0.01 cutoff on eight fixed pilot games. It is a separate small screen, '
            'not another 100-game confirmation. The two-hour review budget saves incomplete games for resumption.</p>'
            '<table><tr><th>Pilot policy</th><th>Complete games</th><th>Mean raw score</th><th>Reached 4096</th><th>Game minutes</th></tr>')
        for depth in (2,3):
            file = root/'evaluation'/f'depth{depth}.json'
            result = json.loads(file.read_text()) if file.exists() else {}
            gamma_pilot += planning_row(f'Gamma 1, up to {depth} steps', result, 8)
        gamma_pilot += '</table><p><a href="gamma1_approximate_depth23_pilot/protocol.json">Reason, budget and process</a></p>'
    seed_pilot = ''
    root = base/'gamma1_seed1_depth2_pilot'
    if (root/'protocol.json').exists():
        file = root/'evaluation/depth2.json'
        result = json.loads(file.read_text()) if file.exists() else {}
        seed_pilot = ('<h2>Does the gamma 1 planner transfer across training seeds?</h2>'
            '<p>Reuse the independently trained seed1 model with exact two-step search, on the first16 '
            'common selection seeds. The seed0 control uses precisely those games. This is a small robustness '
            'pilot with a one-hour resumable review budget, not another100-game result.</p>'
            '<table><tr><th>Training seed</th><th>Complete games</th><th>Mean raw score</th><th>Reached 4096</th><th>Game minutes</th></tr>'
            + planning_row('Seed0 reference subset',json.loads((root/'seed0_control.json').read_text()),16)
            + planning_row('Independent seed1',result,16) + '</table>'
            '<p><a href="gamma1_seed1_depth2_pilot/protocol.json">Reason, budget and process</a></p>')
    afterstate=''
    root=base/'afterstate_two_move_pilots'
    if (root/'comparison.json').exists():
        afterstate=('<h2>Can scalar neural afterstate values make useful, cheaper search leaves?</h2>'
            '<p>These models score boards after a slide. Two explicit moves include one intervening random spawn layer, '
            'then use the scalar afterstate value. Q-network depth labels above use a different leaf convention; '
            'compare the actual computation as well as the number. Same100selectiongames below.</p>'
            '<table><tr><th>Frozen afterstate MLP</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>')
        for label,folder in [('4.19M training moves; two explicit moves','sampled4m'),('29.36M training moves; two explicit moves','sampled29m')]:
            afterstate+=planning_row(label,json.loads((root/folder/'evaluation.json').read_text()),100)
        afterstate+='</table><p>Separate eight-game depth pilot:</p><table><tr><th>Exact moves</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
        for depth,folder in [(2,'pilot_long29m'),(3,'pilot_long29m_depth3')]:
            file=root/folder/'evaluation.json'
            afterstate+=planning_row(str(depth),json.loads(file.read_text()) if file.exists() else {},8)
        afterstate+=('</table><p>The third move gave a modest, inconclusive pilot gain at substantially higher cost. '
            'Its100-game confirmation is deferred while newer gamma1 QR leaves are checked. '
            '<a href="afterstate_two_move_pilots/comparison.json">Training-to-search comparison</a> · '
            '<a href="afterstate_two_move_pilots/depth3_comparison.json">Depth pilot and decision</a></p>')
    symmetry=''
    root=base/'afterstate_symmetry_29m'
    if (root/'comparison.json').exists():
        symmetry=('<h2>Average equivalent board orientations without future-spawn search</h2>'
            '<p>The same frozen29.36M MLP is evaluated on100games, with ordinary afterstate values '
            'or their average over all8rotations/reflections. Both use exact root slides; neither expands future spawns. '
            'Both are rerun on CPU for a matched inference comparison.</p>'
            '<table><tr><th>Value calculation</th><th>Complete games</th><th>Mean score</th><th>Reached2048</th><th>Evaluation seconds</th><th>Fixed replay</th></tr>')
        for mode,label in [('identity','One board view'),('d4_mean','Eight-view average')]:
            s=json.loads((root/f'{mode}.json').read_text())['summary']
            symmetry+=f'<tr><td>{label}</td><td>{s["completed_episodes"]}/100</td><td>{s["mean_score"]:,.2f}</td><td>{s["reaching_2048"]:.0%}</td><td>{s["seconds"]:.2f}</td><td><a href="afterstate_symmetry_29m/replay_{mode}.html">Watch</a></td></tr>'
        symmetry+='</table><p><a href="afterstate_symmetry_29m/comparison.json">Paired scores and limits</a>. A matched training continuation will test differentiable eight-view averaging too.</p>'
    continuation=''
    root=base/'gamma1_continuation_search'
    if (root/'protocol.json').exists():
        continuation=('<h2>Do the new gamma1 continuations improve search?</h2>'
            '<p>Eight predeclared first selection seeds, matched to the original4M reference. '
            'Compare5% versus0.5% exploration continuations from the same original training state. '
            'One- and two-step search are evaluated separately; incomplete suites have no ranked mean.</p>'
            '<table><tr><th>Leaf and search depth</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
            +planning_row('Original4M reference, depth2',json.loads((root/'original4m_depth2_first8.json').read_text()),8))
        for label,folder in [('5% exploration','control005'),('0.5% exploration','low0005')]:
            for depth in (1,2):
                file=root/folder/'evaluation'/f'depth{depth}.json'
                continuation+=planning_row(f'{label}, depth{depth}',json.loads(file.read_text()) if file.exists() else {},8)
        continuation+='</table><p><a href="gamma1_continuation_search/protocol.json">Reason, budget and worker</a></p>'
    continuation_confirmation=''
    root=base/'gamma1_continuation_confirmation'
    if root.exists():
        continuation_confirmation=('<h2>100-game checks of both gamma1 continuations</h2>'
            '<p>The eight-game pilots scored 77,017 with 5% exploration and 63,653 with 0.5%. '
            'The former includes two 8192 games; this small sample cannot establish the ranking. '
            'Both frozen models now use exact two-step planning on the same 100 selection seeds. '
            'The first eight identical completed games are reused, not replayed. '
            'Each worker has a resumable two-hour review budget; partial means stay unranked.</p>'
            '<table><tr><th>Frozen model</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>')
        reference=base/'gamma1_depth2_parallel/depth2.json'
        if reference.exists():
            continuation_confirmation+=planning_row('Original4.19M gamma1 reference',json.loads(reference.read_text()),100)
        for label,folder in [('8.39M; 5% exploration','control005'),('8.39M; 0.5% exploration','low0005')]:
            file=root/folder/'evaluation/depth2.json'
            continuation_confirmation+=planning_row(label,json.loads(file.read_text()) if file.exists() else {},100)
        continuation_confirmation+=('</table><p><a href="gamma1_continuation_search/comparison.json">Paired pilot results and uncertainty</a> · '
            '<a href="gamma1_continuation_confirmation/control005/protocol.json">5% confirmation protocol</a> · '
            '<a href="gamma1_continuation_confirmation/low0005/protocol.json">0.5% confirmation protocol</a></p>')
        if (root/'completed_comparison.json').exists():
            comparison=json.loads((root/'completed_comparison.json').read_text())
            continuation_confirmation+='<p>'+escape(comparison['learning'])+' <a href="gamma1_continuation_confirmation/completed_comparison.json">Completed paired comparison</a></p>'
    symmetric_search=''
    root=base/'afterstate_symmetry_33m_two_move'
    if (root/'protocol.json').exists():
        file=root/'evaluation/evaluation.json'
        result=json.loads(file.read_text()) if file.exists() else {}
        reference=json.loads((base/'afterstate_two_move_pilots/sampled29m/evaluation.json').read_text())
        symmetric_search=('<h2>Can the trained symmetric afterstate model improve cheap planning?</h2>'
            '<p>Two explicit moves with one full spawn layer, using the scalar afterstate network at the boundary. '
            'The newer model averages eight orientations internally and has 4.19M additional training moves. '
            'This compares complete policy recipes, not an isolated symmetry effect. Same 100 selection games. '
            'A bounded 15-minute check temporarily borrows one original deep-search slot and then resumes it.</p>'
            '<table><tr><th>Frozen value model</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
            +planning_row('Ordinary MLP,29.36M training moves',reference,100)
            +planning_row('Trained eight-view MLP,33.55M moves',result,100)
            +'</table><p><a href="afterstate_symmetry_33m_two_move/protocol.json">Reason, budget and worker status</a> · '
            '<a href="afterstate_symmetry_33m_two_move/comparison.json">Paired results and limits</a> · '
            '<a href="afterstate_symmetry_33m_two_move/replay.html">Fixed-seed replay</a></p>')
        pilot=base/'afterstate_symmetry_33m_depth3_pilot'
        reference=base/'afterstate_symmetry_33m_depth3_reference.json'
        if (pilot/'protocol.json').exists() and reference.exists():
            file=pilot/'evaluation/evaluation.json'
            result=json.loads(file.read_text()) if file.exists() else {}
            symmetric_search+=('<p>A separate eight-game pilot tests a third explicit move. '
                'These are the first eight selection seeds by index; the two-move rows are reused '
                'from the completed 100-game evaluation. They are not selected for score.</p>'
                '<table><tr><th>Exact moves</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
                +planning_row('Two moves; reused matching subset',json.loads(reference.read_text()),8)
                +planning_row('Three moves; same frozen value network',result,8)
                +'</table><p><a href="afterstate_symmetry_33m_depth3_pilot/protocol.json">Depth pilot reason and budget</a></p>')
    longer_search=''
    root=base/'afterstate_symmetry_49m_two_move'
    if (root/'evaluation.json').exists():
        newer=json.loads((root/'evaluation.json').read_text())
        parent=json.loads((base/'afterstate_symmetry_33m_two_move/evaluation/evaluation.json').read_text())
        longer_search=('<h2>Further training improves the same two-move planner</h2>'
            '<p>Same symmetric scalar network architecture, gamma1, raw reward and exact two-move search. '
            'Only the learned checkpoint changes after 15.96M additional training moves. Same100CPU selection games; one training lineage.</p>'
            '<table><tr><th>Frozen value model</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
            +planning_row('33.55M lifetime training moves',parent,100)
            +planning_row('49.51M lifetime training moves',newer,100)
            +'</table><p><a href="afterstate_symmetry_49m_two_move/comparison.json">Paired comparison and costs</a> · '
            '<a href="afterstate_symmetry_49m_two_move/audit/replay.html">Fixed replay</a> · '
            '<a href="afterstate_symmetry_49m_two_move/audit/index.html">Move-by-move audit</a></p>')
        pilot=base/'afterstate_symmetry_49m_depth3_pilot'
        if (pilot/'protocol.json').exists():
            reference=json.loads((pilot/'depth2_reference.json').read_text())
            file=pilot/'evaluation/evaluation.json'
            result=json.loads(file.read_text()) if file.exists() else {}
            longer_search+=('<p>A third exact move improved the earlier33.55M model on all8pilot games, '
                'but used about62times the game computation. This follow-up tests the stronger49.51M model '
                'before allocating a full100-game deep search. These are the first8selection seeds by index.</p>'
                '<table><tr><th>Same49.51M model</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
                +planning_row('Two exact moves; reused complete games',reference,8)
                +planning_row('Three exact moves; bounded pilot',result,8)
                +'</table><p><a href="afterstate_symmetry_49m_depth3_pilot/protocol.json">Reason and time budget</a></p>')
    selected_search=''
    root=base/'afterstate_best53m_two_move'
    if (root/'protocol.json').exists():
        file=root/'evaluation/evaluation.json'
        result=json.loads(file.read_text()) if file.exists() else {}
        reference=json.loads((base/'afterstate_symmetry_49m_two_move/evaluation.json').read_text())
        selected_search=('<h2>Does the stronger selected direct policy also help planning?</h2>'
            '<p>The53.18M monitoring-selected low-exploration policy scored42,069 in directMPS play. '
            'This tests it with the same CPUtwo-move planner and100seeds as the49.51M reference. '
            'Direct and planning rankings need not agree. One older search slot is borrowed temporarily and then restored.</p>'
            '<table><tr><th>Frozen model</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
            +planning_row('49.51M reference; CPU',reference,100)
            +planning_row('53.18M selected checkpoint; CPU',result,100)
            +'</table><p><a href="afterstate_best53m_two_move/protocol.json">Reason, budget and restoration status</a> · '
            '<a href="afterstate_best53m_two_move/comparison.json">Paired comparison</a> · '
            '<a href="afterstate_best53m_two_move/audit/replay.html">Fixed replay</a> · '
            '<a href="afterstate_best53m_two_move/audit/index.html">Move-by-move audit</a></p>')
        if (root/'comparison.json').exists():
            comparison=json.loads((root/'comparison.json').read_text())
            selected_search+='<p>'+escape(comparison['interpretation'])+'</p>'
    nonnegative_search=''
    root=base/'afterstate_best53m_nonnegative_two_move'
    if (root/'protocol.json').exists():
        file=root/'evaluation/evaluation.json'
        result=json.loads(file.read_text()) if file.exists() else {}
        reference=json.loads((base/'afterstate_best53m_two_move/evaluation/evaluation.json').read_text())
        nonnegative_search=('<h2>Can impossible negative future scores harm planning?</h2>'
            '<p>The latest fixed replay scored 78,064. Its final board exposed negative value estimates: '
            'the planner preferred an action with 90% immediate death risk over one with no immediate death risk. '
            'A surviving move can still lead to a worse eventual outcome, so this alone does not prove an error. '
            'We test U+(x) = max(U(x), 0) at the learned leaves, retaining every known merge reward. '
            'The weights, CPU device, full spawn probabilities and 100 seeds remain fixed. '
            'The floor changed estimates but none of the 15 audited choices; a full evaluation tests its wider effect.</p>'
            '<table><tr><th>Leaf estimate</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
            +planning_row('Unconstrained; reused complete reference',reference,100)
            +planning_row('Future value floored at zero',result,100)
            +'</table><p><a href="afterstate_best53m_nonnegative_two_move/protocol.json">Reason, budget and live status</a> · '
            '<a href="afterstate_best53m_two_move/audit/nonnegative_counterfactual.json">Same-board counterfactual values</a></p>')
        if (root/'comparison.json').exists():
            comparison=json.loads((root/'comparison.json').read_text())
            nonnegative_search+='<p>'+escape(comparison['learning'])+'</p>'
    mps_search=''
    root=base/'afterstate_symmetry_49m_mps_pilot'
    if (root/'protocol.json').exists():
        mps_search=('<h2>Can the GPU make deeper planning practical?</h2><p>The full planner-call benchmark favoredMPS on seven fixed boards. '
            'This pilot measures complete games and compares two versus three exact moves on the sameMPS backend. '
            'Tiny CPU/MPS differences can change tied actions and entire trajectories; compare depths within the same backend. '
            'The short inference study shares the GPU with one learner, so timings include contention. Eight games are a screen, not a100-game confirmation.</p>'
            '<table><tr><th>MPS policy</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>')
        for depth in (2,3):
            file=root/f'depth{depth}/evaluation.json'
            result=json.loads(file.read_text()) if file.exists() else {}
            mps_search+=planning_row(f'{depth} exact moves; MPS',result,8)
        mps_search+='</table><p><a href="afterstate_symmetry_49m_mps_pilot/protocol.json">Reason and budget</a> · <a href="afterstate_symmetry_49m_two_move/depth3_device_benchmark.json">Full planner-call timings</a></p>'
        confirmation_root=base/'afterstate_symmetry_49m_mps_depth2_confirmation'
        if (confirmation_root/'protocol.json').exists():
            file=confirmation_root/'evaluation.json'
            result=json.loads(file.read_text()) if file.exists() else {}
            mps_search+=('<p>The two-moveMPS reference is now being completed on100selection games, reusing all8finished pilot games. '
                'The three-move100-gameMPS check is deferred while the requested learning comparisons are reviewed: '
                'its eight-game mean gain is uncertain and costs about9times more inference.</p>'
                '<table><tr><th>MPS reference</th><th>Complete games</th><th>Mean score</th><th>Reached4096</th><th>Game minutes</th></tr>'
                +planning_row('Two exact moves;100-game reference',result,100)
                +'</table><p><a href="afterstate_symmetry_49m_mps_depth2_confirmation/protocol.json">Reference protocol and live worker</a></p>')
    page = page.replace('</html>', confirmation + improved + shaped + risk + approximate + gamma + parallel + deeper_confirmation + new_leaf + gamma_pilot + seed_pilot + afterstate + symmetry + continuation + continuation_confirmation + symmetric_search + longer_search + selected_search + nonnegative_search + mps_search + audits + '</html>')
    (base / 'index.html').write_text(page)


if __name__ == '__main__':
    render()
