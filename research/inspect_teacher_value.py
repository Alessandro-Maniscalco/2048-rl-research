"""Show every lookup that contributes to a frozen n-tuple afterstate value."""
import argparse
import json
from pathlib import Path
from html import escape

import numpy as np

from rl2048.agents.ntuple import NTupleAgent,encode
from rl2048.fast2048 import tuple_index,value
from rl2048.game import ACTION_NAMES,move,legal_actions
from rl2048.view import board_html


def breakdown(board,weights,patterns):
    encoded=encode(board);rows=[]
    for pattern in range(len(patterns)):
        contributions=[]
        for orientation in range(8):
            positions=patterns[pattern,orientation]
            index=int(tuple_index(encoded,positions))
            contributions.append(dict(orientation=orientation,index=index,
                tile_exponents=encoded[positions].tolist(),value=float(weights[pattern,index])))
        rows.append(dict(pattern=pattern,lookups=contributions,
                         sum=sum(v['value'] for v in contributions)))
    total=sum(row['sum'] for row in rows)
    np.testing.assert_allclose(total,value(encoded,weights,patterns),atol=1e-6,rtol=1e-12)
    return dict(patterns=len(patterns),orientations=8,lookup_count=len(patterns)*8,
                rows=rows,total_future_points=total,student_target=total/128)


def inspect(checkpoint,replay,board_index,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    frames=json.loads(Path(replay).read_text())['frames'];frame=frames[board_index]
    board=np.asarray(frame['board'],np.int64);teacher=NTupleAgent.load(checkpoint,mmap_mode='r')
    results=[]
    for action in np.flatnonzero(legal_actions(board)):
        after,reward,_=move(board,int(action));details=breakdown(after,teacher.weights,teacher.patterns)
        results.append(dict(action=ACTION_NAMES[action],merge_points=int(reward),board=after.tolist(),
            root_q_points=reward+details['total_future_points'],**details))
    data=dict(checkpoint=str(Path(checkpoint).resolve()),source_replay=str(Path(replay).resolve()),
        board_index=board_index,board=board.tolist(),actions=results,
        explanation='Each action first makes a deterministic afterstate. The target is the SUM of all pattern/orientation table entries, divided by128. It is an estimate of FUTURE points, not points already scored or a guaranteed return. The Transformer receives tiles only; this table is an explanation, not input features.')
    (out/'values.json').write_text(json.dumps(data,indent=2))
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inside the n-tuple teacher</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:20px;background:#faf8ef;color:#544c44}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:right;border-bottom:1px solid #ccc}.scroll{overflow:auto}a{color:#197970}</style>',
        '<h1>The teacher value, lookup by lookup</h1><p>'+escape(data['explanation'])+'</p>',
        board_html(board,frame['score'],f'Before move {board_index}'),
        '<table><tr><th>Action</th><th>Immediate points</th><th>Teacher future points</th><th>Sum: action value</th><th>Student target /128</th></tr>']
    for action in results:
        html.append(f'<tr><td>{action["action"]}</td><td>{action["merge_points"]:,}</td><td>{action["total_future_points"]:,.2f}</td><td>{action["root_q_points"]:,.2f}</td><td>{action["student_target"]:,.3f}</td></tr>')
    html.append('</table>')
    for action in results:
        html.extend([f'<h2>{action["action"]}: {action["patterns"]} tables ×8 views = {action["lookup_count"]} additions</h2>',
            board_html(np.array(action['board']),frame['score']+action['merge_points'],'After slide, before spawn'),
            '<div class="scroll"><table><tr><th>Pattern</th>'+''.join(f'<th>View {v}</th>' for v in range(8))+'<th>Row sum</th></tr>'])
        for row in action['rows']:
            html.append(f'<tr><td>{row["pattern"]}</td>'+''.join(f'<td>{v["value"]:,.2f}</td>' for v in row['lookups'])+f'<td>{row["sum"]:,.2f}</td></tr>')
        html.append(f'</table></div><p>Total: <b>{action["total_future_points"]:,.2f}</b> predicted future points. Student target: <b>{action["student_target"]:,.3f}</b>.</p>')
    html.append('<p><a href="values.json">Exact table indices, tile exponents and unrounded values</a></p></html>')
    (out/'index.html').write_text(''.join(html));return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--board-index',type=int,default=5000);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();inspect(**vars(a))
