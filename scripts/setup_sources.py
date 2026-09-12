"""Fetch pinned optional native sources and apply the published local patches.

Existing directories are left untouched. This does not build native libraries
or download/generate the large endgame tables. See docs/REPRODUCING.md.
"""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT / 'docs/vendor-sources.json').read_text())
    for source in sources:
        destination = ROOT / source['directory']
        if destination.exists():
            print(f'Skipping existing directory: {source["directory"]}')
            continue
        subprocess.run(['git', 'clone', source['url'], str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', source['revision']], check=True)
        patch = ROOT / source['patch']
        if patch.stat().st_size:
            subprocess.run(['git', '-C', str(destination), 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', str(destination), 'apply', str(patch)], check=True)
        print(f'Ready: {source["directory"]} at {source["revision"]}')


if __name__ == '__main__':
    main()
