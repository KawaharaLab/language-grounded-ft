#!/usr/bin/env python3
"""Attach existing paper artifacts without copying or modifying their contents."""
import argparse
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def attach(config, execute=False):
    entries=json.loads(Path(config).read_text())['links']
    prepared=[]
    record=ROOT/'artifacts/local-links.json'
    previous=json.loads(record.read_text())['links'] if record.exists() else []
    destinations={ROOT/e['destination'] for e in entries}
    stale={ROOT/e['destination']:e for e in previous if ROOT/e['destination'] not in destinations}
    if any(a != b and a.is_relative_to(b) for a in destinations for b in destinations):
        raise ValueError('Artifact mount destinations must not nest inside each other')
    for e in entries:
        source=Path(e['source']).expanduser().resolve(strict=True)
        target=ROOT/e['destination']
        if not target.is_relative_to(ROOT/'artifacts') or '..' in Path(e['destination']).parts:
            raise ValueError('Destination must be below artifacts/')
        if target.exists() and not target.is_symlink() and not any(target.is_relative_to(p) for p in stale):
            raise RuntimeError(f'Refusing to replace real files: {target}')
        for parent in target.parents:
            if parent == ROOT: break
            if parent.is_symlink() and parent not in stale:
                raise RuntimeError(f'Refusing to write through a mounted parent: {parent}')
        prepared.append({'source':str(source),'destination':e['destination'],'source_alias':e['source']})
    if execute:
        for target,e in stale.items():
            if target.is_symlink() and target.resolve() == Path(e['source']).resolve():
                target.unlink()
        for e in prepared:
            target=ROOT/e['destination'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_symlink():target.unlink()
            target.symlink_to(e['source'],target_is_directory=Path(e['source']).is_dir())
        (ROOT/'artifacts/local-links.json').write_text(json.dumps({'mode':'external-symlinks','links':prepared},indent=2)+'\n')
    return prepared

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'manifests/local-links.example.json')
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();entries=attach(args.config,args.execute)
    print(json.dumps({'mode':'linked' if args.execute else 'plan','links':len(entries),'copied_bytes':0},indent=2))
if __name__=='__main__':main()
