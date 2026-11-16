#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
files=json.loads((root/'SHA256SUMS.json').read_text())
for name,digest in files.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('KIT_SHA256_PASS',len(files),'files')
