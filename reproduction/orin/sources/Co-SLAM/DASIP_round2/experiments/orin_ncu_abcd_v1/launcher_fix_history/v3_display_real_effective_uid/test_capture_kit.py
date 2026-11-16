"""CPU-only command, variant, immutable-source and report-parser tests."""
from pathlib import Path
import ast,csv,copy,io,json,tempfile,unittest,hashlib,os
from unittest.mock import patch
import capture_queue as q
ROOT=Path(__file__).resolve().parent
class KitTests(unittest.TestCase):
 def setUp(self):self.cfg=json.loads((ROOT/'machine_orin.json').read_text())
 def test_sixteen_unique_capture_paths_and_flags(self):
  jobs=q.cases(self.cfg);self.assertEqual(len(jobs),16);self.assertEqual(len({j['report']for j in jobs}),16)
  for j in jobs:
   cmd=q.ncu_command(j,self.cfg);driver=q.driver_command(j,self.cfg);self.assertEqual(cmd[cmd.index('--launch-count')+1],'500');self.assertEqual(cmd[cmd.index('--launch-skip')+1],'0');self.assertEqual(cmd[cmd.index('--clock-control')+1],'none');self.assertEqual(cmd[cmd.index('--kill')+1],'yes');self.assertEqual([cmd[i+1]for i,x in enumerate(cmd)if x=='--section'],q.SECTIONS);self.assertEqual(driver[driver.index('--frames')+1],'0');self.assertEqual(driver[driver.index('--condition')+1],'BASE');self.assertEqual(driver[driver.index('--ncu-variant')+1],j['variant']);self.assertNotIn('--engineering',driver);self.assertNotIn('--quality',driver);self.assertIn('/bin/bash',cmd)
 def test_both_driver_variant_maps_match_and_no_scientific_change(self):
  for tag,name,manifest,key in [('co','run_coslam_round2.py','RUNTIME_MANIFEST.json','runtime_hashes'),('ec','run_ec_round2.py','EC_RUNTIME_MANIFEST.json','files')]:
   root=ROOT/'runtime'/tag;tree=ast.parse((root/name).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');mapping=next(n.value for n in ast.walk(main)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ncu_variants'for t in n.targets));values=ast.literal_eval(mapping)
   self.assertEqual(values,{k:{'hash':v['hash'],'morton_sort':v['morton_sort']}for k,v in q.VARIANTS.items()})
   parent=json.loads((root/('PARENT_'+manifest)).read_text())
   for rel,h in parent[key].items():
    if rel!=name:self.assertEqual(q.sha(root/rel),h)
   current=json.loads((root/manifest).read_text())
   for rel,h in current[key].items():self.assertEqual(q.sha(root/rel),h)
 def test_mode_and_platform_output_separation(self):
  a=q.cases(self.cfg)[0];cfg=dict(self.cfg,mode='MAXN');b=q.cases(cfg)[0];self.assertNotEqual(a['report'],b['report']);self.assertIn('MAXN',b['report']);self.assertNotIn('--expected-power-mode',q.driver_command(q.cases(dict(self.cfg,platform='RTX',mode='RTX'))[0],dict(self.cfg,platform='RTX',mode='RTX')))
 def fake_raw(self,count):
  f=io.StringIO();w=csv.writer(f);w.writerow(['ID','Process ID','Kernel Name','Context','Section Name','Metric Name','Metric Unit','Metric Value'])
  for i in range(count):
   for section in ['Memory Workload Analysis','Source Counters','GPU Speed Of Light Throughput']:w.writerow([i,42,'tinycudann::kernel_grid',1,section,'test_metric','byte',100])
  return f.getvalue()
 def test_report_requires_500_distinct_launches_and_sections(self):
  with tempfile.TemporaryDirectory(prefix='ncu_report_parse_')as d:
   p=Path(d)/'raw.csv';p.write_text(self.fake_raw(500));self.assertEqual(q.report_count(p)['matched_kernel_launches'],500);self.assertEqual(len(q.verify_sections(p)),3)
   p.write_text(self.fake_raw(499))
   with self.assertRaises(AssertionError):q.report_count(p)
   p.write_text(self.fake_raw(500).replace('Source Counters','Other'))
   with self.assertRaises(AssertionError):q.verify_sections(p)
 def test_existing_gpu_lock_open_omits_create_and_preserves_inode(self):
  original=os.stat('/tmp/dasip_round2_gpu0.lock').st_ino
  actual_open=os.open;calls=[]
  def tracked(path,flags,*args):
   calls.append((path,flags));return actual_open(path,flags,*args)
  with patch.object(q.os,'open',side_effect=tracked):
   with q.open_workload_lock(self.cfg)as lock:self.assertEqual(os.fstat(lock.fileno()).st_ino,original)
  self.assertEqual(len(calls),1);self.assertFalse(calls[0][1]&os.O_CREAT)
  self.assertEqual(os.stat('/tmp/dasip_round2_gpu0.lock').st_ino,original)
 def test_only_exact_verified_idle_display_binaries_are_allowed(self):
  self.assertTrue(q.display_client_allowed('/usr/lib/xorg/Xorg',0))
  self.assertFalse(q.display_client_allowed('/usr/bin/python',0))
  self.assertFalse(q.display_client_allowed('/tmp/Xorg',0))
  self.assertFalse(q.display_client_allowed('/usr/lib/xorg/Xorg',12345))
 def test_cli_never_sets_power_or_clocks(self):
  code=(ROOT/'capture_queue.py').read_text();self.assertNotIn("'nvpmodel','-m'",code);self.assertNotIn("'jetson_clocks'",code)
if __name__=='__main__':unittest.main()
