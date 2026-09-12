"""Live, explicitly frozen-weight comparison of the next-spawn risk check."""
import json
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def render():
    base = Path('runs/research/scaled_transformer/cnn_spawn_safety')
    protocol = read(base/'protocol.json')
    if not protocol:
        return
    status = read(base/'status.json', {})
    comparison = read(base/'comparison.json')
    rows = []
    for name,label in [('baseline','Frozen CNN'), ('safety','Same CNN + next-spawn risk check')]:
        result = read(base/f'{name}.json', {})
        summary = result.get('summary', {})
        mean = summary.get('mean_score')
        rows.append(f'<tr><td>{label}</td><td>{"Complete" if summary.get("complete") else "Pending"}</td>'
            f'<td>{f"{mean:,.0f}" if mean is not None else "—"}</td>'
            f'<td>{summary.get("truncated_episodes", "—")}</td></tr>')
    findings = f'Current phase: {status.get("phase", "pending")} · {status.get("arm", "")}. Both full arms must finish before comparison.'
    if comparison:
        low, high = comparison['paired_game_bootstrap95']
        findings = (f'Paired mean gain {comparison["gain"]:,.0f};95% game-bootstrap interval '
            f'[{low:,.0f}, {high:,.0f}], {comparison["wins"]}/100 wins. '
            f'The check changed {comparison["intervention_counts"]["changed_decisions"]:,} decisions.')
    links = []
    for name in ['baseline','safety']:
        if (base/f'{name}_replay.html').exists():
            links.append(f'<a href="{name}_replay.html">{name.title()} replay</a>')
    replication = base.parent/'cnn_spawn_safety_validation200'
    replication_text = ''
    if (replication/'protocol.json').exists():
        repeat = read(replication/'comparison.json')
        phase = read(replication/'status.json', {})
        replication_text = (f'<h2>Independent200-game replication</h2><p>Current phase: {phase.get("phase","pending")}. '
            'Same frozen CNN and safety rule,200new preselected seeds, four CPU workers,8-board inference chunks. '
            'This execution choice follows a measured CPU batch-size performance cliff.</p>')
        if repeat:
            low,high = repeat['paired_game_bootstrap95']
            replication_text += (f'<p><b>Frozen CNN {repeat["baseline_mean"]:,.0f} → safety {repeat["safety_mean"]:,.0f}.</b> '
                f'Paired gain {repeat["gain"]:,.0f},95% interval [{low:,.0f}, {high:,.0f}], '
                f'{repeat["wins"]}/200wins and {repeat["ties"]}ties. '
                'These seeds exclude the original100games. Both arms use the same CPU execution; '
                'changing batch shape can change floating-point rounding, so the two studies are reported separately.</p>')
        replication_text += ('<p><a href="../cnn_spawn_safety_validation200/protocol.json">Replication protocol</a> · '
            '<a href="../cnn_spawn_safety_validation200/status.json">Status</a> · '
            '<a href="../cnn_spawn_safety_validation200/comparison.json">Completed comparison</a></p>')
        if (replication/'replay_verification.json').exists():
            replication_text += ('<p>Replays use the first preselected replication seed,8958800. Their score, length and largest tile match the batched results; '
                'every safety replay action has minimum immediate risk. This one game is not the average improvement. '
                '<a href="../cnn_spawn_safety_validation200/baseline_replay.html">Baseline replay</a> · '
                '<a href="../cnn_spawn_safety_validation200/safety_replay.html">Safety replay</a> · '
                '<a href="../cnn_spawn_safety_validation200/replay_verification.json">Verification</a>.</p>')
    page = f'''<!doctype html><html><meta charset="utf-8"><title>Can avoiding immediate death improve2048?</title>
    <style>body{{font:18px system-ui;line-height:1.6;max-width:950px;margin:40px auto;padding:20px;color:#203040}}
    table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#245ab5}}</style>
    <a href="../index.html">All research</a><h1>Can avoiding immediate death improve2048?</h1>
    <p>The exact same pretrained CNN plays100fresh paired seeds on CPU, with frozen weights.
    One arm follows its usual highest legal logit. The other chooses its highest logit among legal moves
    with the lowest probability that the next random tile ends the game.</p><p><b>{findings}</b></p>
    <table><tr><th>Policy</th><th>Status</th><th>Mean raw score</th><th>Truncated games</th></tr>{''.join(rows)}</table>
    <p>Risk(s,a) = Σ<sub>spawn z</sub> p(z|slide(s,a)) ×1[the resulting board has no legal moves].</p>
    <p>If a slide leaves two or more empty cells, one empty cell remains after spawning, so immediate death risk is zero.
    With exactly one empty cell, test a2and a4there, weighted90%and10%. Two targeted tests compare this shortcut
    with exhaustive enumeration and reproduce the avoidable10%risk in the actual GPT game.</p>
    <p>This is an explicit inference intervention using known game rules. Immediate survival can conflict with eventual points,
    so success is measured by complete-game scores. No teacher, reward shaping, corner rule or new network training is used.</p>
    <p>Both arms compute risk statistics, so reported wall times include that diagnostic overhead and are not the unmodified CNN inference speed.</p>
    <p>{' · '.join(links)} · <a href="protocol.json">Protocol</a></p>{replication_text}</html>'''
    (base/'index.html').write_text(page)


if __name__ == '__main__':
    render()
