"""Sample read-only CPU/GPU clock and temperature telemetry during the suites."""
import argparse
import csv
import datetime
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    try:
        return path.read_text().strip()
    # Some sysfs entries are unavailable through the restricted filesystem
    # proxy; keep the column empty rather than aborting the experiment monitor.
    except (OSError, UnicodeError, TypeError):
        return ''


def active_jobs():
    active = []
    for name in ('coslam_screen','ec_screen','coslam_full','ec_full','ec_repeat_full','coslam_repeat_full'):
        paths = list((ROOT / 'results/execution_experiments/logs').glob('*/' + name + '.log'))
        # Retain compatibility with older queues that redirect logs at root.
        paths.append(ROOT / ('results_execution_' + name + '.log'))
        for path in paths:
            if not path.exists():
                continue
            lines = read(path).splitlines()
            if lines and lines[-1].startswith('START '):
                active.append(str(path.relative_to(ROOT)) + ': ' + lines[-1])
    return json.dumps(active)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--until', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=1.)
    args = parser.parse_args()
    paths = sorted(Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq'))
    paths += sorted(Path('/sys/class/devfreq').glob('*/cur_freq'))
    paths += sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['utc','unix_seconds','active_jobs', *map(str,paths)])
        writer.writeheader()
        while True:
            record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          unix_seconds=time.time(), active_jobs=active_jobs())
            record.update({str(p):read(p) for p in paths})
            writer.writerow(record)
            stream.flush()
            if args.until.exists():
                break
            time.sleep(args.interval)
    print('TELEMETRY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
