"""Allow declared artifact mounts while rejecting legacy live-code dependencies.

Loaded only when tools/run.py adds this directory to PYTHONPATH. Approved artifact roots are listed in artifacts/local-links.json. Historical
metadata paths remain translated by loaders; mounts may point to the existing SSD data. This audit hook does not rewrite filesystem operations.
"""
import os
import sys
import json
from pathlib import Path

LEGACY_PREFIXES = (
    '/home/user/Genesis', '/home/user/PatchTST', '/home/user/ur3_robotiq_ros2',
    '/mnt/nvme1/user/data/Genesis', '/mnt/nvme1/user/data/PatchTST',
    '/mnt/nvme1/user/data/ur2_robotiq_ros2',
)

manifest = Path(__file__).resolve().parents[2]/'artifacts/local-links.json'
ALLOWED_PREFIXES = ()
if manifest.exists():
    links = json.loads(manifest.read_text())['links']
    ALLOWED_PREFIXES = tuple(str(p) for e in links for p in (Path(e['source']), Path(e.get('source_alias', e['source']))))

def _guard(event, arguments):
    if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes)):
        return
    path = os.path.abspath(os.fsdecode(arguments[0]))
    allowed = any(path == prefix or path.startswith(prefix + '/') for prefix in ALLOWED_PREFIXES)
    if not allowed and any(path == prefix or path.startswith(prefix + '/') for prefix in LEGACY_PREFIXES):
        raise PermissionError('Legacy research path access blocked; resolve it into FT_WORKSPACE_ROOT first: ' + path)

if os.environ.get('FT_ENFORCE_PORTABLE_PATHS') == '1':
    sys.addaudithook(_guard)
