"""Exact source manifests, per-file integrity and study-wide provenance guards."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import config
from scripts.Validation import validate_pruning8 as audit

ROOT=Path(__file__).resolve().parents[1]


def mapping(names, suffix=''):
    return {name:hashlib.sha256((name+suffix).encode()).hexdigest() for name in names}


class SourceManifestTests(unittest.TestCase):
    def write_archive(self, root, names):
        hashes=mapping(names)
        for name in names:
            target=root/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(name)
        (root/'source_hashes.json').write_text(json.dumps(hashes))
        return hashes
    def test_only_exact_fifteen_or_seventeen_source_sets_are_accepted(self):
        for manifest,names in audit.SOURCE_MANIFESTS.items():
            with self.subTest(manifest=manifest),TemporaryDirectory() as temporary:
                path=Path(temporary)
                hashes=self.write_archive(path,names)
                self.assertEqual(audit.source_manifest_name(hashes),manifest)
                self.assertEqual(audit.inspect_sources(path),hashes)
                for edited in (dict(hashes,**{'extra.py':'0'*64}),
                               {k:v for k,v in hashes.items() if k!='coslam.py'}):
                    (path/'source_hashes.json').write_text(json.dumps(edited))
                    with self.assertRaises(ValueError):audit.inspect_sources(path)
                (path/'source_hashes.json').write_text(json.dumps(hashes))
                (path/'source/coslam.py').write_text('changed archived source')
                with self.assertRaisesRegex(ValueError,'archived source hash differs'):
                    audit.inspect_sources(path)
    def test_bad_digest_is_rejected_without_installation_fallback(self):
        with TemporaryDirectory() as temporary:
            path=Path(temporary);hashes=self.write_archive(path,audit.CLEAN_REQUIRED_SOURCES)
            hashes['coslam.py']='not-a-digest'
            (path/'source_hashes.json').write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError,'invalid SHA-256'):
                audit.inspect_sources(path)
    def test_pair_rejects_mixed_manifests_and_different_file_bytes(self):
        clean=mapping(audit.CLEAN_REQUIRED_SOURCES)
        variants=(mapping(audit.FROZEN_REQUIRED_SOURCES),dict(clean,**{'coslam.py':'0'*64}))
        for other in variants:
            with patch.object(audit,'inspect_run',side_effect=[({}, {}, clean),({}, {}, other)]):
                with self.assertRaisesRegex(ValueError,'dense and candidate source differ'):
                    audit.inspect_pair(Path('/missing/control'),Path('/missing/selected'),'Replica','office0',2,0)
    def test_study_rejects_manifest_mixing_or_same_manifest_changed_bytes_across_seeds(self):
        clean=mapping(audit.CLEAN_REQUIRED_SOURCES)
        variants=(mapping(audit.FROZEN_REQUIRED_SOURCES),dict(clean,**{'coslam.py':'0'*64}))
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            for family in ('controls','selected'):
                for seed in (0,1):
                    path=root/family/f'Replica_office0_seed{seed}'
                    path.mkdir(parents=True);(path/'summary.json').write_text('{}')
            def result(seed,hashes):
                row=dict(dataset='Replica',scene='office0',seed=seed,
                    dense_ate_cm=1.,selected_ate_cm=1.,dense_tracking_ms=10.,selected_tracking_ms=9.,
                    dense_ba_ms=10.,selected_ba_ms=7.)
                return row,{},hashes,{},len(hashes)*2,dict(device='cpu',torch_version='test',python_version='test')
            for other in variants:
                with patch.object(audit,'SCENES',(('Replica','office0',2),)),\
                     patch.object(audit,'inspect_pair',side_effect=[result(0,clean),result(1,other)]):
                    with self.assertRaisesRegex(ValueError,'Runtime source must be identical'):
                        audit.evaluate(root,seeds=(0,1))
    def test_clean_configuration_omits_inert_options_and_reaches_normal_diagnostics(self):
        class ReachedDiagnostics(Exception):pass
        cfg=config.load_config(str(ROOT/'configs/Replica/office0.yaml'))
        cfg['grid'].update(hash='CoherentPrime',hash_size=19,morton_sort=False)
        cfg['data']['output']='fixture'
        cfg['ba_experiment']=cfg['tracking_experiment']=dict(variant='baseline')
        for name in ('ba_bucket','tracking_bucket','tracking_naive_pruning','tracking_depth_spatial'):
            cfg.setdefault(name,{})['enabled']=False
        cfg['tracking'].update(iter=10,best=True)
        cfg['sample_pruning']=dict(enabled=True,policy='dense',seed=0,
                                  tracking_pose_selection='native')
        chosen=copy.deepcopy(cfg)
        chosen['sample_pruning'].update(policy='ray_cohort',stages=['Tracking','Bundle_Adjustment'],
            cohort_tracking='ray_cycle',tracking_cycle_groups=2,tracking_anchor_every=3,
            tracking_pose_selection='dense_anchors',cohort_count=4,recent_keyframes=2,
            cohort_transition_keyframes=2,tv_fraction=.125)
        hashes=mapping(audit.CLEAN_REQUIRED_SOURCES)
        summary=dict(device='cpu',torch_version='test',python_version='test',map_optimizer_eps={},ray_database={})
        with patch.object(audit,'inspect_run',side_effect=[(summary,cfg,hashes),(summary,chosen,hashes)]),\
             patch.object(audit,'inspect_cohorts',side_effect=ReachedDiagnostics):
            with self.assertRaises(ReachedDiagnostics):
                audit.inspect_pair(Path('/missing/control'),Path('/missing/selected'),'Replica','office0',2,0)
        chosen['sample_pruning']['ray_estimator']='self_normalized'
        with patch.object(audit,'inspect_run',side_effect=[(summary,cfg,hashes),(summary,chosen,hashes)]):
            with self.assertRaisesRegex(ValueError,'omit removed experimental options'):
                audit.inspect_pair(Path('/missing/control'),Path('/missing/selected'),'Replica','office0',2,0)
    def test_supplemental_evidence_binds_archives_and_installed_sources_without_relabeling(self):
        frozen=mapping(audit.FROZEN_REQUIRED_SOURCES)
        clean=mapping(audit.CLEAN_REQUIRED_SOURCES,'clean')
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'equivalence.json'
            evidence=dict(schema_version=1,frozen_source_hashes=frozen,clean_source_hashes=clean,
                          cpu_equivalence_passed=True)
            path.write_text(json.dumps(evidence))
            result=audit.cleanup_equivalence_metadata(path,frozen,dict(source_hashes=clean))
            self.assertTrue(result['source_hash_bindings_verified'])
            self.assertTrue(result['supplemental_only'])
            self.assertFalse(result['replaces_archived_gpu_sources'])
            self.assertFalse(result['clean_gpu_quality_or_timing_rerun'])
            changed=dict(clean,**{'coslam.py':'0'*64})
            with self.assertRaisesRegex(ValueError,'currently installed'):
                audit.cleanup_equivalence_metadata(path,frozen,dict(source_hashes=changed))
            with self.assertRaisesRegex(ValueError,'archived runtime'):
                audit.cleanup_equivalence_metadata(path,changed,dict(source_hashes=clean))


if __name__=='__main__':unittest.main(verbosity=2)
