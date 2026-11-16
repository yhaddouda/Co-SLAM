"""CPU-only tests for the queue amendment and lock-preserving normal-boundary handoff."""
from pathlib import Path
import copy,importlib.util,json,os,sys,tempfile,types,unittest
from unittest.mock import patch
A=Path(__file__).resolve().parent;R=A.parents[1]
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class ExecCaptured(Exception):pass
class AmendmentTests(unittest.TestCase):
 def test_bindings_are_execution_identical(self):
  old={j['id']:j for j in json.loads((A/'original/resolved/ALL_LOCAL_TARGETS.json').read_text())['jobs']};new=json.loads((A/'resolved/ALL_LOCAL_TARGETS.json').read_text())
  self.assertEqual([p['count']for p in new['phases']],[48,56,26,5]);self.assertEqual(len(new['jobs']),135)
  for j in new['jobs']:
   for key in ['argv','output','runtime','source_sha256','datadir','expected_native_frames','seed','hash_size','driver_condition']:self.assertEqual(j.get(key),old[j['id']].get(key))
   self.assertEqual(j['seed'],0)
   if j['condition']=='MC':self.assertEqual(j['dataset'],'Replica')
   if j['condition']=='MC_DP'and j['dataset']!='Replica':self.assertNotIn('MC',j['control_ids']);self.assertIn('BASE',j['control_ids']);self.assertIn('DP',j['control_ids']);self.assertIn('MC',j['unavailable_comparisons'])
 def test_handoff_exec_retains_fd_without_signals_or_children(self):
  h=load('tested_normal_boundary_handoff',A/'normal_boundary_handoff.py')
  with tempfile.TemporaryDirectory(prefix='orin_queue_amend_test_')as td,open('/tmp/dasip_round2_gpu0.lock','a')as lock:
   t=Path(td);rr=t/'controller';aa=rr/'amendment';(rr/'process_receipts').mkdir(parents=True);aa.mkdir();fd=lock.fileno();os.set_inheritable(fd,True)
   expected=dict(pid=os.getpid(),start_ticks=1,boot_id='test');(aa/'AMENDMENT_REQUEST.json').write_text(json.dumps(dict(supervisor_identity=expected,retained_ids=['DP'],files={})));(rr/'process_receipts/DP.json').write_text(json.dumps(dict(returncode=0,child_identity=dict(pid=999999))))
   fake=types.ModuleType('__main__');fake.__file__=str(rr/'supervisor.py');fake.identity=lambda pid:expected;fake.support=types.SimpleNamespace(still_alive=lambda p:False);fake.freeze=lambda:None;fake.mode=lambda:None;fake.clocks=lambda:None;fake.idle=lambda three:[];fake.now=lambda:'test';fake.PYTHON=Path(sys.executable);fake.h=h;fake.fd=fd
   exec("def main():\n s={'active_child_identity':None}\n j={'id':'DP'}\n fd=globals()['fd']\n h.at_normal_boundary()\n",fake.__dict__)
   with patch.object(h,'A',aa),patch.object(h,'R',rr),patch.dict(sys.modules,{'__main__':fake}),patch.object(h.os,'execve',side_effect=ExecCaptured)as exe,patch.object(h.os,'kill',side_effect=AssertionError('No signals permitted')):
    with self.assertRaises(ExecCaptured):fake.main()
   self.assertEqual(exe.call_count,1);argv=exe.call_args.args[1];self.assertEqual(argv[argv.index('--adopt-owned-lock-fd')+1],str(fd));self.assertTrue(os.get_inheritable(fd));self.assertEqual(json.loads((aa/'NORMAL_BOUNDARY_HANDOFF.json').read_text())['status'],'NORMAL_CHILD_COMPLETED_EXEC_HANDOFF')
 def test_no_handoff_from_amended_controller(self):
  h=load('tested_handoff_ignore',A/'normal_boundary_handoff.py')
  with patch.object(h.os,'execve',side_effect=AssertionError('Unexpected exec')):h.at_normal_boundary()
if __name__=='__main__':unittest.main()
