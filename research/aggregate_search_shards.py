"""Validate independent game shards before reporting a combined score."""
import hashlib
import json
from pathlib import Path


def aggregate(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    expected_hash = hashlib.sha256((Path(protocol['checkpoint']) / 'network/policy.pt').read_bytes()).hexdigest()
    episodes = []
    modified = []
    errors = []
    for shard in protocol['shards']:
        directory = root / shard['directory']
        config_file = directory / 'config.json'
        result_file = directory / f'depth{protocol["depth"]}.json'
        if not config_file.exists() or not result_file.exists():
            continue
        try:
            config = json.loads(config_file.read_text())
            result = json.loads(result_file.read_text())
        except json.JSONDecodeError:
            errors.append(f'{directory.name}: JSON write in progress or unreadable')
            continue
        if (config['checkpoint_sha256'] != expected_hash or config['seed_start'] != shard['seed_start']
                or config['games'] != shard['games'] or config['depths'] != [protocol['depth']]
                or config['probability_cutoff'] != protocol['probability_cutoff']):
            raise ValueError(f'{directory.name}: shard does not match the declared experiment')
        allowed = set(range(shard['seed_start'],shard['seed_start']+shard['games']))
        if any(row['seed'] not in allowed for row in result['episodes']):
            raise ValueError(f'{directory.name}: unexpected game seed')
        episodes.extend(result['episodes'])
        modified.append(result_file.stat().st_mtime)
    seeds = [row['seed'] for row in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Game seeds overlap between shards')
    expected = set(range(protocol['seed_start'],protocol['seed_start']+protocol['games']))
    complete = set(seeds) == expected and all(e['complete'] and not e['truncated'] for e in episodes)
    result = dict(depth=protocol['depth'],requested_games=protocol['games'],complete=complete,
        episodes=sorted(episodes,key=lambda e:e['seed']),read_errors=errors,
        completed_games=sum(e['complete'] for e in episodes),
        mean_score=sum(e['score'] for e in episodes)/len(episodes) if complete else None,
        worker_game_seconds=sum(e['seconds'] for e in episodes),
        wall_seconds=max(modified)-protocol['started_epoch'] if complete else None,
        timing_note='Worker game seconds sum concurrent work; wall seconds cover launch through the last saved game result, including any pauses.',
        checkpoint_sha256=expected_hash)
    file = root / f'depth{protocol["depth"]}.json'
    temporary = file.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False))
    temporary.replace(file)
    return result
