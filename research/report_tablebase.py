"""Separate complete-game averages, selected records and genuine GPT play."""
import json
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def render():
    base = Path('runs/research/tablebase_search')
    rows, bests = [], []
    for name,label,prefix in [('validation100', 'Frozen depth-8 validation', 'validation_depth8'),
                              ('record500', 'Depth-5 record-seeking batch', 'record_depth5')]:
        summary = read(base/name/'summary.json', {})
        games = read(base/name/'games.json', [])
        complete = [g for g in games if g.get('complete')]
        if not complete:
            continue
        best = max(complete, key=lambda g:g['score'])
        folder = f'{prefix}_seed{best["seed"]}'
        bests.append((best, folder))
        finished = summary.get('complete', summary.get('all_games_complete', False))
        mean = summary.get('mean_score', summary.get('completed_game_mean'))
        n = len(complete)
        planned = summary.get('planned_games', summary.get('planned_attempts'))
        failures = sum(not g.get('complete') for g in games)
        status = 'Complete' if finished else 'In progress · provisional mean'
        rows.append(f'<tr><td>{label}</td><td>{status}</td><td>{n}/{planned}</td><td>{failures}</td>'
            f'<td>{mean:,.0f}</td><td><a href="{folder}/replay.html">{best["score"]:,}</a></td>'
            f'<td>{sum(g["max_tile"] >= 65536 for g in complete)}</td></tr>')
    if not bests:
        return
    best, folder = max(bests, key=lambda x:x[0]['score'])
    checked = (base/folder/'independent_recheck.json').exists()
    screens = '<h2>What the recent search changes established</h2>'
    probability = read(base/'probability_screen/summary.json')
    if probability:
        screens += ('<p>At fixed depth5, following lower-probability spawn branches did not demonstrate a gain in12paired games. '
            'The wider setting averaged657,090 versus701,442, with a paired difference interval spanning zero, '
            'and cost2.48times as much per game. Different seeds from the100-game validation: '
            'the701k screening mean is not a new validated average. '
            '<a href="probability_screen/summary.json">Saved screen</a>.</p>')
    gate = read(base/'lookup_gate_screen/summary.json')
    if gate:
        screens += (f'<p>Another12-pair screen let general search override tuple10 recommendations when their '
            f'restricted subgoal probability was at most1%. Mean score changed only {gate["mean_paired_difference"]:.2f} points: '
            '11ties and one28-point difference. This threshold produced no meaningful improvement. '
            'The isolated binary and tables were unchanged during evaluation. '
            '<a href="lookup_gate_screen/protocol.json">Reason and controls</a> · '
            '<a href="lookup_gate_screen/summary.json">Results</a>.</p>')
    gate10 = read(base/'lookup_gate10_screen/comparison.json')
    if gate10:
        low,high=gate10['paired_game_bootstrap95']
        screens += (f'<p><b>The 10% confidence override was rejected.</b> On 12 new paired games the mean fell '
            f'from {gate10["by_setting"]["original"]["mean_score"]:,.0f} to {gate10["by_setting"]["gate"]["mean_score"]:,.0f}. '
            f'Paired change {gate10["mean_paired_difference"]:,.0f}, 95% interval [{low:,.0f}, {high:,.0f}]. '
            'At the first divergence in the first preselected game, lookup chose up at9.9%subgoal probability; '
            'general search chose down. Both merged 12 points and had zero immediate death risk. '
            'A difficult subgoal can still be the better long-term plan; a low probability alone does not justify overriding it. '
            '<a href="lookup_gate10_screen/first_divergence.json">The actual board and alternatives</a>.</p>')
    deeper = read(base/'depth8_vs9_fixed_chance/summary.json')
    if (base/'depth8_vs9_fixed_chance/protocol.json').exists():
        screens += ('<p><b>Next search comparison:</b> depth 8 versus 9 with the same 1/4096 chance cutoff, '
            'the same cache-safe binary and unchanged table hierarchy. Eight preselected paired games, six CPU workers. '
            'Extra look-ahead is isolated from probability pruning; partial means are provisional. '
            '<a href="depth8_vs9_fixed_chance/protocol.json">Protocol and limits</a> · '
            '<a href="depth8_vs9_fixed_chance/summary.json">Live results</a>.</p>')
        if deeper:
            screens += (f'<p>Finished {deeper["finished_attempts"]}/16 attempts; '
                f'{deeper["active"]} active. '+('Both endpoints complete.' if deeper['complete'] else
                'The paired comparison is still incomplete.')+'</p>')
            if deeper['complete']:
                low,high=deeper['paired_game_bootstrap95']
                screens += (f'<p>Depth 8 mean {deeper["by_depth"]["8"]["provisional_mean_score"]:,.0f}; '
                    f'depth 9 mean {deeper["by_depth"]["9"]["provisional_mean_score"]:,.0f}. '
                    f'Paired gain {deeper["mean_paired_difference"]:,.0f}, 95% interval [{low:,.0f}, {high:,.0f}]. '
                    'This is an eight-pair screen, not a new 100-game validation.</p>')
    page = f'''<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="60">
    <title>2048 high-score research</title><style>body{{font:18px system-ui;line-height:1.5;max-width:1100px;margin:40px auto;padding:0 20px;color:#202b38}}
    table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #ddd}}a{{color:#245ab5}}strong{{font-size:1.2em}}</style>
    <h1>2048 high-score research</h1><p><strong>Best completed game: {best['score']:,} points · tile {best['max_tile']:,}</strong></p>
    <p><a href="{folder}/replay.html">Watch the highest-scoring replay</a> · {best['length']:,} moves · standard two-tile start · natural termination.
    All recorded slides, merges, spawns and score increments match our Python rules.
    {'A separate full replay recheck also passed.' if checked else ''}</p>
    <p>This is an external search player with solved-pattern tables, not a newly trained neural model or GPT choosing its moves.
    It is our project record, not a verified world record. A selected maximum depends on the number of attempts.</p>
    <table><tr><th>Experiment</th><th>Status</th><th>Complete games</th><th>Incomplete attempts</th><th>Mean score</th><th>Best game</th><th>65536 games</th></tr>{''.join(rows)}</table>
    <p>The depth-8 setting was selected on a separate 10-game depth comparison and frozen before 100 fresh games.
    The 500-game depth-5 batch tests a different idea: its cheaper decisions permit more record attempts per hour.
    It uses an isolated fix for ambiguous high-rank cache keys. All attempts are retained; partial means remain provisional.</p>
    <h2>How it decides</h2><p>When a recognized large-tile arrangement permits a table lookup,
    choose a move using a computed probability of completing a smaller merge-building goal.
    Otherwise, consider legal slides, model random spawns, and search future choices using
    a programmed board-quality evaluation that favors descending tile arrangements and mergeable neighbors.
    Chance is averaged; the player cannot choose where the next tile appears.</p>
    <p>At a decision node: choose the legal move with maximum estimated value.<br>
    At a random-spawn node: value = Σ probability × value after observing that spawn.<br>
    Table values are subgoal-success probabilities; search leaf values are heuristic board scores.
    The reported game score is the actual sum of merge rewards, a different quantity from those decision values.</p>
    {screens}<h2>Genuine GPT game</h2><p>GPT scored 12,252 in 722 moves, reaching 1024.
    Same-seed frozen references scored 544 (random), 9,232 (our 48,474-mean Transformer), and 361,440 (root n-tuple teacher).
    One game cannot establish average superiority. The final GPT move had 10% immediate death risk, while other legal choices had 0%.
    <a href="../gpt6_direct/seed8948000/replay.html">Watch GPT's actual decisions</a>.</p>
    <p><a href="../scaled_transformer/cnn_reinforce.html">Current neural research: learning from actual own-game returns</a> ·
    <a href="../scaled_transformer/cnn_spawn_safety/index.html">Frozen CNN safety improvement</a> ·
    <a href="../scaled_transformer/index.html">All neural experiments</a> ·
    <a href="https://github.com/macroxue/2048-ai">External player source and provenance</a></p></html>'''
    (base/'index.html').write_text(page)


if __name__ == '__main__':
    render()
