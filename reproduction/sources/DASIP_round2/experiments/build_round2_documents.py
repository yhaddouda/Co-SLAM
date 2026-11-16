#!/usr/bin/env python3
"""Build Round 2 PDFs using a workspace-local font cache, never system changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    cache = root / 'build/tex_cache'
    for name in ('var', 'config'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, TEXMFVAR=str(cache / 'var'), TEXMFCONFIG=str(cache / 'config'))
    fontmap = cache / 'var/fonts/map/pdftex/updmap/pdftex.map'
    if not fontmap.exists():
        with (cache / 'updmap.log').open('w') as log:
            subprocess.run(['updmap', '--user'], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([os.sys.executable, str(root / 'experiments/prepare_round2_placeholder.py')], check=True)
    jobs = [(root / 'docs/methods', 'CONSERVATIVE_VARIANT_EXPLAINED.tex', ['-outdir='+str(root / 'build/explanation')]),
            (root / 'paper_round2_placeholder', 'paper_round2_placeholder.tex', [])]
    for directory, filename, extra in jobs:
        logpath = root / 'build/logs' / ('build_' + Path(filename).stem + '.log')
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with logpath.open('w') as log:
            result = subprocess.run(['latexmk', '-pdf', '-g', '-interaction=nonstopmode', '-halt-on-error',
                                     *extra, filename], cwd=directory, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('LaTeX build failed; inspect ' + str(logpath))
    shutil.copy2(root / 'build/explanation/CONSERVATIVE_VARIANT_EXPLAINED.pdf', root / 'docs/methods/CONSERVATIVE_VARIANT_EXPLAINED.pdf')
    outputs = [root / 'docs/methods/CONSERVATIVE_VARIANT_EXPLAINED.pdf', root / 'paper_round2_placeholder/paper_round2_placeholder.pdf']
    manifest = {}
    for path in outputs:
        info = subprocess.check_output(['pdfinfo', str(path)], text=True)
        pages = int(next(line.split(':', 1)[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
        manifest[str(path.relative_to(root))] = dict(pages=pages, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   bytes=path.stat().st_size)
    (root / 'metadata/DOCUMENT_BUILD_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
