"""Summarize completed experiments; never rank partial games as complete games."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parents[1] / 'runs/research/deeper_q'


def main():
    search = {}
    for depth in [2, 3]:
        p = B / f'screen/depth{depth}.json'
        if p.exists(): search[depth] = json.loads(p.read_text())
    games = []
    for p in sorted((B / 'depth4_games').glob('game*/depth4.json')):
        games.extend(json.loads(p.read_text())['episodes'])
    by_seed = {g['seed']:g for g in games}
    for p in sorted((B / 'depth4_retries').glob('game*/depth4.json')):
        for g in json.loads(p.read_text())['episodes']:
            if g['complete'] or g['seed'] not in by_seed: by_seed[g['seed']] = g
    games = list(by_seed.values())
    search[4] = dict(episodes=games, requested_games=8, complete=len(games)==8 and all(g['complete'] for g in games))
    search[4]['mean_score'] = float(np.mean([g['score'] for g in games])) if search[4]['complete'] else None
    (B / 'search_results.json').write_text(json.dumps(search, indent=2))
    lines = ['# Does deeper lookahead help?', '',
      'This returns to the locally trained 16-input, 256-wide, two-hidden-layer Q-network that benefited from two-step search. It does not use the newly imported PPO CNN.', '',
      '## Experiment 1: same Q-network, deeper exact search', '',
      'Inputs: current board tile exponents. Process: enumerate legal slides and every possible random spawn, take the best legal future action and average chance outcomes. Output: four lookahead action values; argmax selects the move.', '',
      'Keep the Q weights, gamma 0.99, corner/snake potential reward, and game seeds fixed. Search depth counts explicit move/spawn layers beyond the learned Q leaf. The leaf still estimates rewards beyond the explicit horizon.', '',
      'F_0(s,a) = Q_old(s,a)',
      'F_d(s,a) = sum_over_spawns p(s_next | s,a) [r_learning + gamma * max_legal F_(d-1)(s_next,a_next)]',
      'Terminal continuation is zero. r_learning = raw_merge_score/128 + 2*(gamma*Phi(s_next)-Phi(s)); terminal Phi is zero.', '',
      '| Lookahead | Completed / requested games | Mean raw score | Reached 2048 |',
      '|---|---:|---:|---:|']
    for depth, r in search.items():
        finished = sum(e['complete'] for e in r['episodes'])
        score = f"{r['mean_score']:,.2f}" if r['complete'] else 'Pending / incomplete'
        rate = f"{np.mean([e['max_tile']>=2048 for e in r['episodes']]):.0%}" if r['complete'] else '—'
        lines.append(f"| {depth} | {finished} / {r['requested_games']} | {score} | {rate} |")
    lines += ['', 'These are eight-game screening comparisons on seeds 8600000–8600007, not a 100-game final benchmark. Paired game-seed intervals do not measure variation between training seeds.']
    comparisons = {}
    for shallow, deep in [(2,3),(3,4)]:
        if search.get(shallow,{}).get('complete') and search.get(deep,{}).get('complete'):
            a = {e['seed']:e['score'] for e in search[shallow]['episodes']}
            delta = np.array([e['score']-a[e['seed']] for e in search[deep]['episodes']])
            radius = 1.96*delta.std(ddof=1)/np.sqrt(len(delta))
            comparisons[f'{deep}_minus_{shallow}'] = dict(mean=float(delta.mean()), approximate_95_percent_ci=[float(delta.mean()-radius),float(delta.mean()+radius)], wins=int((delta>0).sum()))
            lines.append(f'Depth {deep} minus {shallow}: paired mean {delta.mean():,.2f}; approximate 95% interval [{delta.mean()-radius:,.2f}, {delta.mean()+radius:,.2f}]; wins {(delta>0).sum()}/{len(delta)} games.')
    (B / 'paired_search_comparison.json').write_text(json.dumps(comparisons, indent=2))
    confirmation = {}
    for depth in [2,3]:
        episodes = []
        for p in sorted((B / 'confirmation').glob(f'worker*/depth{depth}.json')):
            episodes.extend(json.loads(p.read_text())['episodes'])
        complete = len(episodes)==100 and all(e['complete'] for e in episodes)
        confirmation[depth] = dict(episodes=episodes, complete=complete,
            mean_score=float(np.mean([e['score'] for e in episodes])) if complete else None)
    (B / 'confirmation_results.json').write_text(json.dumps(confirmation, indent=2))
    lines += ['', '### Separate 100-game confirmation', '',
              'Both frozen policies use seeds 8601000–8601099. No student training uses these games.', '',
              '| Depth | Complete games | Mean score |', '|---|---:|---:|']
    for depth, r in confirmation.items():
        score = f"{r['mean_score']:,.2f}" if r['complete'] else 'Pending / incomplete'
        lines.append(f"| {depth} | {sum(e['complete'] for e in r['episodes'])}/100 | {score} |")
    if all(r['complete'] for r in confirmation.values()):
        a = {e['seed']: e['score'] for e in confirmation[2]['episodes']}
        d = np.array([e['score']-a[e['seed']] for e in confirmation[3]['episodes']])
        radius = 1.96*d.std(ddof=1)/10
        paired = dict(mean=float(d.mean()), approximate_95_percent_ci=[float(d.mean()-radius),float(d.mean()+radius)], wins=int((d>0).sum()))
        (B / 'paired_confirmation.json').write_text(json.dumps(paired, indent=2))
        lines.append(f'Paired depth-three minus depth-two difference: {d.mean():,.2f}, approximate 95% interval [{d.mean()-radius:,.2f}, {d.mean()+radius:,.2f}]; wins {(d>0).sum()}/100 games.')
    lines += ['', 'Learning: the initial three-step screen was promising but uncertain; the separate 100-game confirmation supports an improvement over two steps. Four-step results remain an eight-game screen. Do not infer that more depth always helps.', '']
    p = B / 'depth5_pilot/depth5.json'
    if p.exists():
        five = json.loads(p.read_text())
        e = five['episodes'][0]
        lines += ['## Experiment: exact five-step pilot', '',
                  'The pilot used seed 8600000, a 30-minute wall-clock cap, MPS, exact state merging and a 40-million-outcome limit per decision. No probability pruning was used.', '']
        if e['complete']:
            lines.append(f"One complete game scored {e['score']:,}; one game cannot establish superiority.")
        else:
            lines.append(f"The game remained unfinished at the time cap: {e['length']:,} moves, partial score {e['score']:,}, maximum tile {e['max_tile']:,}. This is not a final score and is excluded from ranking. Board and RNG state are saved for an explicitly requested continuation.")
    pruned = {}
    for depth in [4,5]:
        episodes = []
        for p in sorted((B/'pruned').glob(f'depth{depth}_seed*/depth{depth}.json')):
            episodes.extend(json.loads(p.read_text())['episodes'])
        complete = len(episodes)==4 and all(e['complete'] for e in episodes)
        pruned[depth] = dict(episodes=episodes, complete=complete,
                            mean_score=float(np.mean([e['score'] for e in episodes])) if complete else None)
    (B/'pruned_results.json').write_text(json.dumps(pruned,indent=2))
    lines += ['', '## Experiment: make five-step search cheaper', '',
        'Both variants use a chance-path probability cutoff of 0.0001 and seeds 8600000–8600003. Unlikely branches stop at the learned Q estimate; their probability mass is retained. This is approximate, up-to-depth search, not exact depth-five search. The maximum incoming path probability is used when identical states merge. Decisions in these runs query one root board at a time.', '',
        '| Maximum depth | Complete games | Mean raw score |', '|---|---:|---:|']
    for depth, r in pruned.items():
        score = f"{r['mean_score']:,.2f}" if r['complete'] else 'Incomplete; no mean'
        lines.append(f"| {depth} | {sum(e['complete'] for e in r['episodes'])}/4 | {score} |")
    lines += ['', 'These runs shared the remaining five-step-stage deadline (about 409 seconds each, in parallel). Depth five did not finish its games, so no score advantage is established. The more aggressive 0.001 cutoff was timing-probed on three boards only; no complete-game evaluation was run for it.', '',
        'Learning: selective expansion reduces cost substantially, but a five-step score comparison remains unfinished within this pilot budget. All experiment processes stopped at the end of this stage. Neither time-limited partial games nor timing probes are evidence of stronger play.', '',
      '## Experiment 2: teach the MLP the lookahead values', '',
      'Each student starts from identical original Q weights. Training inputs are 1,536 boards sampled from 12 separate depth-two teacher games; 512 boards from four other games check label generalization. Every depth uses the same boards, batches, initialization and 2,000 Adam updates (batch 256, learning rate 1e-4) on MPS.', '',
      'Targets: y_d(s,a) = F_d(s,a), with frozen original Q leaves. The student receives only the current board. It learns to approximate the search results; it does not execute those future moves internally.', '',
      'The first objective fits both Q levels and action gaps: mean_legal[e^2 + 10*(e - mean_legal(e))^2], where e = Q_student - y_d. A controlled alternative adds an explicit preferred-action objective: cross_entropy(Q_student/0.1, argmax(y_d)) + 0.01*first_objective.', '',
      '| Direct-action model, no search | Objective | Mean score over 100 games |',
      '|---|---|---:|']
    for folder, label in [('distillation','Q regression'),('ranking_students','Action ranking + weak Q regression')]:
        for depth in [2,3]:
            p = B / folder / f'evaluation_student_depth{depth}.json'
            if p.exists():
                s = json.loads(p.read_text())['summary']
                lines.append(f"| Student taught depth {depth} | {label} | {s['mean_score']:,.2f} |")
    p = B / 'distillation/evaluation_original_direct.json'
    if p.exists(): lines.append(f"| Original Q-network | No new training | {json.loads(p.read_text())['summary']['mean_score']:,.2f} |")
    lines += ['', 'All these direct policies use the same 100 complete validation games, seeds 8620000–8620099. These games are separate from teacher-label data. Because we compared an alternative loss after viewing the first results, treat this as validation, not an untouched final test.', '']
    strong = B / 'stronger_students'
    if (strong/'evaluation_original_direct.json').exists():
        lines += ['### Transfer into the stronger pretrained MLP', '',
            'A second starting point uses our existing 1,136,644-parameter relational MLP (1,024-wide hidden layers), pretrained locally for 243,002,880 transitions. Both depth variants retain the same student initialization, board dataset, targets, 2,000 updates and ranking objective; learning rate is 1e-5. This comparison tests teacher depth within a stronger parent. It is not an isolated network-width ablation against the small-student experiments.', '',
            '| Strong MLP variant | Mean score over the same 100 validation games |', '|---|---:|']
        for name, label in [('original_direct','Original stronger parent'),('student_depth2','Trained on depth-two targets'),('student_depth3','Trained on depth-three targets')]:
            s = json.loads((strong/f'evaluation_{name}.json').read_text())['summary']
            lines.append(f"| {label} | {s['mean_score']:,.2f} |")
        lines += ['', 'This transfer also reduces playing strength. Here, the three-step student is worse than the matched two-step student. More lookahead in the labels does not guarantee a better learned policy. No student is promoted, and this failing offline recipe was not extended to depth-four or depth-five targets.', '']
    lines += [
      'Learning: three-step student scores are higher than two-step student scores in these runs, but both transfers substantially degrade the original direct policy. Changing the loss helps somewhat and still fails to recover the parent. Q regression reproduces only 53–56% of teacher actions even on training boards, versus 50–54% on held-out boards. Ranking raises training agreement to 72–76%, but held-out agreement is only 53–57%. Thus fitting the training targets is itself difficult; the ranking variant also shows a generalization gap. A small offline sample and mismatch between teacher-visited and student-visited states are plausible limitations, not proven causes. Keep the parent; do not promote a student.', '',
      '## Compute and exactness', '',
      'Both CPU and MPS were timed. CPU was slightly faster for this small network at depth three. Four games can run concurrently on CPU; student optimization uses the Apple GPU. Timing under concurrent workload is not an equal-wall-clock algorithm comparison.', '',
      'The planner merges identical 128-bit board keys at the same depth, preserving all probabilities and legal actions. Hash-based merging gave identical benchmark values and sped up repeated-state processing compared with sorting. The first four depth-four games started with sorting; subsequent games and necessary retries use mathematically equivalent hash merging. Several time-limited attempts could not be ranked; their logs remain preserved. Final results use completed games only. CPU and MPS were used as recorded per run, so these are not matched elapsed-compute comparisons. Exact runs use cutoff zero. Only the separately labeled probability-cutoff experiment stops at Q estimates earlier on unlikely branches. No spawn sampling or tile downgrading is used.', '',
      'Initial depth-five tests hit a two-million-outcome resource limit. Increasing it to twenty million allowed all three timing boards: approximately 2.9, 12.9 and 10.5 seconds per move, with 6.7, 17.2 and 14.8 million expanded outcomes. Those are timing probes, not complete-game score results. No depth-five student has been trained.', '',
      'A later exact legality optimization checks adjacent cells instead of constructing successor boards just to test move availability. It matches full movement checks on 1,000 test boards, and the depth-five benchmark Q values are unchanged. Combined with hash merging and MPS, the three timing probes took about 1.2, 5.8 and 3.8 seconds. The five-step game was resumed from its saved board and RNG state with this optimization; its original 30-minute deadline was retained.', '',
      '## Files and commands', '',
      '- rl2048/agents/deep_q_planning.py: exact search, state merging and explicit resource exceptions.',
      '- research/deeper_q_experiment.py: complete seeded game comparisons and saved planner checkpoints.',
      '- research/distill_deeper_q.py: teacher data collection, search targets, student optimization and direct-policy evaluation.',
      '- tests/test_deep_q_planning.py: agreement with the old two-step planner, independent depth-three/four/five recursion, terminal behavior, resource limits and save/reload.',
      '- runs/research/deeper_q/: configurations, logs, board datasets, targets, checkpoints and per-game metrics.', '',
      'uv run python research/deeper_q_experiment.py --depths 2 3 --games 8',
      'uv run python research/distill_deeper_q.py', '',
      'Use a new output directory when starting another experiment so existing results remain reviewable.']
    report = '\n'.join(lines)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    complete = [(d,r) for d,r in search.items() if r['complete']]
    axes[0].bar([str(d) for d,r in complete], [r['mean_score'] for d,r in complete], color='#167565')
    axes[0].set(title='Exact search: 8 matched games', xlabel='Explicit search depth', ylabel='Mean raw score')
    names, scores = ['Original'], []
    p = B / 'distillation/evaluation_original_direct.json'
    if p.exists(): scores.append(json.loads(p.read_text())['summary']['mean_score'])
    for folder, tag in [('distillation','Regression'),('ranking_students','Ranking')]:
        for depth in [2,3]:
            p = B / folder / f'evaluation_student_depth{depth}.json'
            if p.exists():
                names.append(f'{tag}\ndepth {depth}')
                scores.append(json.loads(p.read_text())['summary']['mean_score'])
            p = B / folder / f'training_depth{depth}.json'
            if p.exists():
                h = json.loads(p.read_text())['history']
                axes[2].plot([x['update'] for x in h], [x['validation_action_agreement'] for x in h], label=f'{tag}, depth {depth}')
    if scores: axes[1].bar(names, scores, color=['#8c857e']+['#167565']*(len(scores)-1))
    axes[1].set(title='Small MLPs: 100 validation games', ylabel='Mean raw score')
    axes[1].tick_params(axis='x', labelsize=8)
    axes[2].set(title='Held-out teacher-action agreement', xlabel='Gradient updates', ylabel='Fraction of actions matched', ylim=(0,1))
    axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(axis='y', alpha=.15)
    fig.savefig(B / 'comparison.png', dpi=150)
    plt.close(fig)
    (B / 'journal.md').write_text(report)
    (B / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Deeper Q lookahead</title><style>body{font:17px/1.6 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#203130;background:#fafaf6}pre{white-space:pre-wrap;font:inherit}a{color:#167565}img{width:100%}</style><h1>Does deeper lookahead help?</h1><p><a href="search_results.json">Search results</a> · <a href="paired_search_comparison.json">Paired comparisons</a> · <a href="journal.md">Experiment notes</a></p><img src="comparison.png" alt="Search scores, student scores, and teacher-action agreement"><pre>'+html.escape(report)+'</pre>')
    print(json.dumps({d:r['mean_score'] for d,r in search.items()}))


if __name__ == '__main__': main()
