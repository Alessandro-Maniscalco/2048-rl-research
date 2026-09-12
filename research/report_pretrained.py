"""Build the local report and a median-validation-score example replay."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from rl2048.agents.neural import NeuralAgent
from rl2048.view import save_replay


ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'runs/research/pretrained_cnn'


def main():
    torch.set_num_threads(2)
    rows = list(csv.DictReader((B / 'finetune/progress.csv').open()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout='constrained')
    x = [int(r['transitions']) for r in rows]
    for ax, key, title in zip(axes, ['mean_score_100', 'value_loss', 'entropy'],
                              ['Training score (recent completed games)', 'Critic MSE loss', 'Policy entropy'], strict=True):
        ax.plot(x, [float(r[key]) if key != 'mean_score_100' or float(r[key]) > 0 else float('nan') for r in rows])
        ax.set(title=title, xlabel='Environment transitions')
        ax.grid(alpha=.2)
    axes[1].set_yscale('log')
    fig.savefig(B / 'training.png', dpi=150)
    plt.close(fig)

    episodes = json.loads((B / 'original_validation.json').read_text())['episodes']
    episode = sorted(episodes, key=lambda e: e['score'])[len(episodes) // 2]
    agent = NeuralAgent.load(B / 'original')
    agent.display_name = 'Pretrained CNN PPO — median validation example'
    replay = save_replay(agent, B / 'replay.html', seed=episode['seed'])
    assert replay['score'] == episode['score'], (replay, episode)

    report = (ROOT / 'research/PRETRAINED_OPTIONS.md').read_text()
    (B / 'journal.md').write_text(report)
    table = ''
    for name in ['original', 'finetuned']:
        s = json.loads((B / f'{name}_test.json').read_text())['summary']
        table += f'<tr><td>{name}</td><td>{s["mean_score"]:,.0f}</td><td>{s["reaching_2048"]:.0%}</td><td>{s["max_tile_distribution"].get("8192",0)}%</td></tr>'
    page = f'''<!doctype html><meta charset="utf-8"><title>Pretrained neural 2048</title>
<style>body{{font:17px/1.6 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#203130;background:#fafaf6}}h1{{line-height:1.2}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}img{{width:100%}}pre{{white-space:pre-wrap;font:15px/1.6 system-ui}}a{{color:#167565}}</style>
<h1>A pretrained neural player, now running locally</h1>
<p>1.45 million parameters · current board only · CNN + two MLP heads · Apple GPU fine-tuning</p>
<p><b>Result: no demonstrated improvement from the short fine-tuning pilot.</b> Original preserved as the preferred checkpoint.</p>
<p>100 held-out games per frozen checkpoint, identical seeds, legal argmax actions, no planning.</p>
<table><tr><th>Checkpoint</th><th>Mean raw score</th><th>Reached 2048</th><th>Reached 8192</th></tr>{table}</table>
<p><a href="replay.html">Watch a median validation example of the pretrained player</a> · <a href="comparison_test.json">Paired test comparison</a> · <a href="journal.md">Full experiment notes</a></p>
<img src="training.png" alt="Training scores, critic loss and policy entropy over one million transitions">
<p>Training curves describe sampled training games, not independent evaluation performance. Early incomplete rollouts do not have a completed-game mean.</p>
<pre>{html.escape(report)}</pre>'''
    (B / 'index.html').write_text(page)
    print('Replay verified against batch evaluation:', replay)


if __name__ == '__main__':
    main()
