"""Resolve pre-organization document/bundle names without changing archives."""
import json
from pathlib import Path


def local_path(base, relative):
    base = Path(base).resolve()
    relative = str(relative)
    if Path(relative).is_absolute():
        try:
            relative = str(Path(relative).relative_to(base))
        except ValueError:
            return Path(relative)
    locations = json.loads((base/'metadata/FILE_LOCATIONS.json').read_text())['moves']
    for old,new in sorted(locations.items(), key=lambda item:len(item[0]), reverse=True):
        if relative==old or relative.startswith(old+'/'):
            return base/(new+relative[len(old):])
    return base/relative
