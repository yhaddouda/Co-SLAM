"""CPU-only import and deferred recovery integration tests; never reads timing CSVs."""
from pathlib import Path
import importlib.util,sys,types,tempfile,json,unittest
from unittest.mock import patch
A=Path(__file__).resolve().parent;R=A.parents[1]
class LoaderTests(unittest.TestCase):
 def test_actual_relocated_auditor_import(self):
  spec=importlib.util.spec_from_file_location('relocated_native_auditor_test',R/'native_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);self.assertTrue(callable(m.audit));self.assertEqual(Path(sys.modules['supervisor'].__file__).resolve(),A/'supervisor.py')
 def test_completed_result_recovery_at_normal_boundary(self):
  import supervisor as shim
  with tempfile.TemporaryDirectory(prefix='orin_deferred_audit_test_')as td:
   r=Path(td);a=r/'amendment';a.mkdir();(r/'process_receipts').mkdir();jobid='ScanNet_scene0059_DP_H12_seed0';reason=json.loads((A/'original_failures'/(jobid+'.json')).read_text())['error'];(r/'process_receipts'/(jobid+'.json')).write_text(json.dumps(dict(returncode=0,elapsed_seconds=3.0,utc='finish',child_identity=dict(pid=999999))))
   fake=types.ModuleType('__main__');fake.__file__=str(a/'supervisor_amended.py');fake.shim=shim;fake.job=dict(id=jobid,status='failed',reason=reason);fake.audit=lambda j:dict(status='PASS',summary=dict(frames=1807));fake.support=types.SimpleNamespace(still_alive=lambda p:False);fake.freeze=lambda:None;fake.mode=lambda:None;fake.clocks=lambda:None;fake.idle=lambda:[];fake.now=lambda:'test';fake.identity=lambda pid:dict(pid=pid);fake.update=lambda s:None
   def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
   fake.save=save;exec("def main():\n s={'jobs':[job],'active_child_identity':None}\n shim._recover_finished_audit()\n",fake.__dict__)
   with patch.object(shim,'A',a),patch.object(shim,'R',r),patch.dict(sys.modules,{'__main__':fake}):fake.main()
   self.assertEqual(fake.job['status'],'done');self.assertEqual(json.loads((a/'AUDIT_LOADER_RECOVERY.json').read_text())['status'],'PASS');self.assertEqual(fake.job['resolved_controller_error'],reason)
if __name__=='__main__':unittest.main()
