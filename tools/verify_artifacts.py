#!/usr/bin/env python3
"""Verify required recorded artifact hashes, without loading or executing them."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    manifest=json.loads((ROOT/'manifests/robot-artifacts.json').read_text())
    results=[]
    for entry in manifest['artifacts']:
        if entry.get('blocking') is False: continue
        path=ROOT/entry['destination']
        actual=digest(path) if path.is_file() else None
        results.append({'path':entry['destination'],'expected':entry['sha256'],'actual':actual,'ok':actual is not None and actual==entry['sha256']})
    report={'results':results,'passed':all(e['ok'] for e in results)}
    print(json.dumps(report,indent=2))
    return not report['passed']

if __name__=='__main__': sys.exit(main())
