#!/usr/bin/env python3
"""Run Python in a contained component environment and generated source tree."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component', choices=['simulation','learning'])
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args=parser.parse_args()
    arguments=args.args[1:] if args.args[:1]==['--'] else args.args
    python=ROOT/'.venvs'/args.component/'bin/python'
    if not python.exists(): parser.error(f'Environment missing. Run tools/setup_env.sh {args.component}')
    workspace=ROOT/'workspace'
    cwd=workspace/('Genesis' if args.component=='simulation' else 'PatchTST/PatchTST_self_supervised')
    env=os.environ.copy()
    env.update(FT_ENFORCE_PORTABLE_PATHS='1',GS_CACHE_FILE_PATH=str(ROOT/'.cache/genesis'),XDG_CACHE_HOME=str(ROOT/'.cache'),FT_REPO_ROOT=str(ROOT),FT_WORKSPACE_ROOT=str(workspace),HF_HOME=str(ROOT/'artifacts/huggingface'),PYTHONDONTWRITEBYTECODE='1')
    model=ROOT/'artifacts/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130'
    if model.is_dir(): env.setdefault('FT_TEXT_MODEL',str(model))
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'tools/runtime'),str(workspace/'Genesis'),str(workspace/'PatchTST/PatchTST_self_supervised'),str(cwd)])
    return subprocess.call([str(python),*arguments],cwd=cwd,env=env)

if __name__=='__main__': sys.exit(main())
