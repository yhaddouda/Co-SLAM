"""Record dependency versions and tiny-cuda-nn bindings without initializing CUDA."""
import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import site
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    versions = {}
    for name in ('torch', 'pytorch3d', 'tinycudann', 'tiny-cuda-nn', 'numpy', 'scipy',
                 'opencv-python', 'opencv-python-headless', 'open3d', 'mathutils',
                 'matplotlib', 'PyYAML', 'tqdm', 'imageio', 'trimesh'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    artifacts = {}
    for root in site.getsitepackages():
        path = Path(root)
        files = list(path.glob('tinycudann_bindings/*.so'))
        files += list(path.glob('tinycudann/*.py'))
        for file in files:
            artifacts[str(file)] = dict(bytes=file.stat().st_size,
                sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    modules = {}
    for name in ('torch', 'pytorch3d', 'tinycudann', 'tinycudann_bindings', 'mathutils'):
        spec = importlib.util.find_spec(name)
        modules[name] = spec.origin if spec else None
        # Editable installs place sources/bindings outside site-packages.
        if spec and name in ('tinycudann', 'tinycudann_bindings'):
            for location in spec.submodule_search_locations or []:
                files = list(Path(location).glob('*.py')) + list(Path(location).glob('*.so'))
                for file in files:
                    artifacts[str(file)] = dict(bytes=file.stat().st_size,
                        sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    import cv2
    versions['cv2_runtime'] = cv2.__version__
    result = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version, executable=sys.executable, architecture=platform.machine(),
        packages=versions, module_origins=modules, tinycudann_artifacts=artifacts,
        scope='Installed distribution metadata and binding/source hashes; CUDA not initialized. '
              'Missing distribution metadata does not alone imply a module is unavailable.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('ENVIRONMENT_CAPTURED ' + str(args.output))


if __name__ == '__main__':
    main()
