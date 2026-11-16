"""Tracking optimizer updates must not mutate saved camera poses."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'coslam.py').read_text())
scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CoSLAM')
namespace = dict(torch=torch)
for name in ('get_pose_param_optim', 'predict_current_pose'):
    node = next(node for node in scene.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'coslam.py', 'exec'), namespace)


class PoseStorageTests(unittest.TestCase):
    def test_translation_optimizer_does_not_change_input_poses(self):
        poses = torch.eye(4)[None].repeat(2, 1, 1)
        poses[:, :3, 3] = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
        before = poses.clone()
        slam = SimpleNamespace(config=dict(tracking=dict(lr_rot=.01, lr_trans=.01)),
                               matrix_to_tensor=lambda rotation: rotation[:, :, 0])
        rotation, translation, optimizer = namespace['get_pose_param_optim'](slam, poses, mapping=False)
        translation.sum().backward()
        optimizer.step()
        torch.testing.assert_close(poses, before, rtol=0, atol=0)
        self.assertFalse(torch.equal(translation, before[:, :3, 3]))
        self.assertTrue(rotation.is_leaf and translation.is_leaf)

    def test_first_prediction_and_nonconstant_prediction_own_their_storage(self):
        for frame, constant_speed in ((1, True), (2, False)):
            stored = torch.eye(4)
            slam = SimpleNamespace(est_c2w_data={frame - 1: stored}, device='cpu')
            prediction = namespace['predict_current_pose'](slam, frame, constant_speed)
            prediction[:3, 3] += .1
            torch.testing.assert_close(stored, torch.eye(4), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
