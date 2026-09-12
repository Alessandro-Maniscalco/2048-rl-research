"""Explain delayed reward with independent, same-policy horizon comparisons."""
from html import escape
import json
from pathlib import Path


def render():
    root=Path('runs/research/endgame_tablebase')
    out=root/'long_rollout_validation'
    if not (out/'horizon_analysis.json').exists():return
    s=json.loads((out/'summary.json').read_text())
    h=json.loads((out/'horizon_analysis.json').read_text())
    panels=[]
    for case,horizon in zip(s['cases'],h['cases']):
        rows=[]
        for a,ah in zip(case['actions'],horizon['actions']):
            ci=a['paired_bootstrap95']
            means=ah['mean_points_by_horizon']
            label=a['name']+(' — recorded move' if a['name']==case['recorded_action'] else '')
            rows.append(f'<tr><td>{escape(label)}</td><td>{means["8"]:,.1f}</td>'
                f'<td>{means["128"]:,.1f}</td><td>{means["512"]:,.1f}</td>'
                f'<td>{a["paired_difference_from_recorded"]:+,.1f}<br><small>[{ci[0]:+,.1f}, {ci[1]:+,.1f}]</small></td>'
                f'<td>{a["terminated"]} / {a["horizon_capped"]}</td></tr>')
        panels.append(f'<h2>Before move {case["source_move"]:,}</h2>'
            f'<p>The recorded move was <strong>{case["recorded_action"]}</strong>. Each action received 128 independent fresh future streams.</p>'
            '<div class="scroll"><table><thead><tr><th>First move</th><th>8-move points</th><th>128-move points</th>'
            '<th>512-move points</th><th>512-move difference from recorded<br><small>paired 95% interval</small></th>'
            '<th>Ended / reached 512 moves</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>')
    (out/'index.html').write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Why short-horizon advice can fail</title>
    <style>body{font:17px system-ui;line-height:1.55;max-width:1160px;margin:36px auto;padding:0 22px;color:#20313e}
    h1{line-height:1.2}.note{padding:18px;background:#fff0ca;border-radius:10px}.scroll{overflow-x:auto}
    table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:11px 8px;border-bottom:1px solid #ddd;text-align:left}
    img{width:100%;height:auto}a{color:#245ab5}code{background:#eef2f5;padding:5px}</style></head><body>
    <h1>Small immediate risk can preserve a much larger payoff</h1>
    <p class="note"><strong>Independent validation on two recorded boards.</strong> The original player's moves have the highest
    estimated 512-move returns on 128 fresh streams per action. Shorter-horizon averages prefer different moves on both boards.
    These are conditional simulations, not new complete-game scores or proof of an optimal policy.</p>
    <p>The boards come from the verified <a href="../work_table_ablation/screen/seed8982012_tables1/replay.html">1,357,916-point game</a>.
    We force one legal direction, then let the existing one-million-work hybrid continue under fresh random spawns.
    The old formation tables and compact search determine subsequent actions; no new network was trained.</p>
    <h2>What the numbers mean</h2>
    <p><code>Q̂π_H(s,a) = (1/N) Σᵢ Σₜ₌₀ᴴ⁻¹ rᵢ,ₜ₊₁</code></p>
    <p><strong>s</strong> is the recorded board; <strong>a</strong> is the forced first direction;
    <strong>π</strong> is the existing base player; <strong>r</strong> is actual merge points; <strong>H</strong> is the number of moves included;
    <strong>N</strong> is 128 future streams. Rewards after a natural game end are zero. All points beyond H are excluded.</p>
    <p>Every horizon uses the <strong>same policy and the same simulated trajectories</strong>, so changing H isolates the effect of
    waiting for delayed rewards. The earlier exact eight-step planner optimized all future actions instead; that is a different value.</p>
    <img src="horizons.png" alt="Mean raw points at horizons 8, 32, 128 and 512 for each first direction at two recorded boards">
    '''+''.join(panels)+'''
    <h2>What changed our interpretation</h2>
    <p>Before move 29,720, down earns more within eight moves; up earns much more over 512. Before move 44,666,
    left earns more within eight moves; right earns much more over 512. Large merges occur after the short window closes.</p>
    <p>From the second board, right built another 32,768 tile in 18 of 128 simulated futures, compared with 6 after left,
    5 after down and none after up. These are fresh hypothetical futures from a saved position, excluded from full-game records.
    The original game's right move happened to receive a fatal spawn. Avoidable immediate risk does not, by itself, show a mistake.</p>
    <h2>What this does not establish</h2>
    <p>These estimates use a fixed base player, finite samples and zero value after 512 moves. They are not optimal Q values,
    and do not guarantee an online rollout policy would improve full-game scores. Pairwise intervals are exploratory and unadjusted
    for multiple comparisons. Two selected boards do not represent the distribution of all games.</p>
    <p>The 16-stream discovery sample and 128-stream validation use disjoint seeds and are reported separately.
    Every one of the 1,024 validation trajectories passed its exact seeded move/reward/spawn audit. Time- or memory-limited failures
    would block the comparison; natural ends and intentional 512-move limits are both valid samples of this finite-horizon quantity.</p>
    <p>This follows the idea of evaluating candidate actions through simulated continuation with a base policy, described in
    <a href="https://ocw.mit.edu/courses/6-231-dynamic-programming-and-stochastic-control-fall-2015/resources/mit6_231f15_lec9/">MIT's rollout lecture</a>.
    Exact policy-improvement results do not automatically apply to these finite sampled estimates.</p>
    <p><a href="protocol.json">Frozen validation protocol</a> · <a href="summary.json">All estimates and intervals</a> ·
    <a href="horizon_analysis.json">Same-policy horizon analysis</a> · <a href="horizons.svg">Export figure</a> ·
    <a href="../long_rollout_probe/summary.json">Separate 16-stream discovery</a> ·
    <a href="../work_table_ablation/index.html">Completed formation-table comparison</a> · <a href="../index.html">All research</a></p>
    </body></html>''')


if __name__=='__main__':render()
