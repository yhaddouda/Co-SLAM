"""Single-JSON bridge to the validated Orin host ownership checker."""
import json
import sys
import argparse
from local_support import idle_sample

parser = argparse.ArgumentParser()
parser.add_argument('--expected-mode', default='MAXN')
args = parser.parse_args()
try:
    print(json.dumps(dict(idle=True, evidence=idle_sample(args.expected_mode))))
except Exception as error:
    print(json.dumps(dict(idle=False, error=repr(error))))
    print(repr(error), file=sys.stderr)
    sys.exit(1)
