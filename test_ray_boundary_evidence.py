import unittest
import numpy as np
from ray_boundary_evidence import score_patch, rays_from_prediction, make_boundary_scorer


class RayEvidenceTests(unittest.TestCase):
    def test_boundary_adapter_respects_multiview_support(self):
        K = np.array([[2.,0,1],[0,2,1],[0,0,1]])
        data = dict(depth=np.full((3,3,3),2.), confidence=np.ones((3,3,3)),
                    intrinsics=np.repeat(K[None],3,axis=0),
                    extrinsics=np.repeat(np.eye(4)[None],3,axis=0))
        frame = dict(origin_world=[0,0,0],e1=[1,0,0],e2=[0,0,1])
        scorer = make_boundary_scorer(data,frame,[0,1,0],[-1,1],.05,stride=1)
        r = scorer([-1,2],[1,2])
        self.assertFalse(r['free_space_contradiction'])
        self.assertEqual(r['supporting_frames'],3)
        self.assertGreater(r['occupied_fraction'],0)
        self.assertAlmostEqual(sum(r[k] for k in ('free_space_fraction','occupied_fraction','unobserved_fraction')),1)

    def score(self, endpoints, **kw):
        p = np.asarray(endpoints, float)
        return score_patch(np.zeros_like(p), p, np.arange(len(p)),
                           [0, 0, 2], [[1, 0, 0], [0, 1, 0]], [[-1, -1], [1, 1]], .05, **kw)

    def test_observed_occluded_conflicting_and_unseen(self):
        r = self.score([[0, 0, 2], [0, 0, 1], [0, 0, 3], [6, 0, 2], [0, 0, -2], [1, 0, 0]])
        self.assertEqual(r['counts'], dict(observed_support=1, free_space_conflict=1, occluded=1, unobserved=3))

    def test_occluded_region_is_not_confirmed_wall(self):
        r = self.score([[0, 0, 1]])
        self.assertEqual(r['status'], 'no_visible_evidence')
        self.assertIsNone(r['mean_per_view_conflict_fraction'])

    def test_same_world_geometry_under_rotation_translation_and_scale(self):
        p = np.array([[0, 0, 2], [0, 0, 1], [0, 0, 3.]])
        Q = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
        b = np.array([100., -40, 7])
        r = score_patch(np.zeros_like(p)+b, 7*p@Q.T+b, np.arange(3),
                        7*np.array([0, 0, 2])@Q.T+b, np.array([[1.,0,0],[0,1,0]])@Q.T,
                        np.array([[-1.,-1],[1,1]])*7, .35)
        self.assertEqual(r['counts'], self.score(p)['counts'])

    def test_each_view_has_equal_weight_not_each_pixel(self):
        p = np.array([[0.,0,3]]*100+[[0.,0,2]])
        r = score_patch(np.zeros_like(p), p, np.array([0]*100+[1]), [0,0,2],
                        [[1,0,0],[0,1,0]], [[-1,-1],[1,1]], .05)
        self.assertEqual(r['mean_per_view_conflict_fraction'], .5)

    def test_prediction_rays_reproject_correctly_with_rotated_camera(self):
        R = np.array([[0.,0,1],[1,0,0],[0,1,0]])
        pose = np.c_[R, [1.,2,3]]
        data = dict(depth=np.full((1,2,2),2.), confidence=np.ones((1,2,2)),
                    intrinsics=np.eye(3)[None], extrinsics=pose[None])
        o,p,ids = rays_from_prediction(data, stride=1)
        np.testing.assert_allclose(p@R.T+pose[:,3], [[0,0,2],[2,0,2],[0,2,2],[2,2,2]])
        np.testing.assert_allclose(o@R.T+pose[:,3], 0)
        self.assertEqual(ids.tolist(), [0]*4)

    def test_no_valid_depth(self):
        data = dict(depth=np.zeros((1,2,2)), confidence=np.ones((1,2,2)),
                    intrinsics=np.eye(3)[None], extrinsics=np.eye(4)[None])
        self.assertEqual(rays_from_prediction(data)[0].shape, (0,3))

    def test_duplicated_source_frames_are_not_independent_views(self):
        data = dict(depth=np.full((3,2,2),2.), confidence=np.ones((3,2,2)),
                    intrinsics=np.repeat(np.eye(3)[None],3,axis=0),
                    extrinsics=np.repeat(np.eye(4)[None],3,axis=0), source_indices=np.array([7,7,7]))
        o,p,ids = rays_from_prediction(data,stride=1)
        self.assertEqual(np.unique(ids).tolist(),[7])
        r = score_patch(o,p,ids,[0,0,2],[[1,0,0],[0,1,0]],[[-3,-3],[3,3]],.05)
        self.assertEqual(r['visible_view_count'],1)
        data['component_ids'] = np.array([0,0,1])
        with self.assertRaisesRegex(ValueError,'disconnected'):
            rays_from_prediction(data,stride=1)

    def test_local_opening_does_not_contaminate_other_patch_cells(self):
        # Left ray stops on wall; right ray crosses candidate wall through an opening.
        p = np.array([[-.5, .5, 2], [.75, .75, 3], [-.25, -.25, 1]])
        r = score_patch(np.zeros_like(p), p, [0, 0, 1], [0,0,2],
                        [[1,0,0],[0,1,0]], [[-1,-1],[1,1]], .05, grid_size=2)
        cells = {tuple(c['cell']): c for c in r['spatial_grid']['cells']}
        self.assertEqual(cells[(0,1)]['conflict_fraction_equal_view'], 0.)
        self.assertEqual(cells[(1,1)]['conflict_fraction_equal_view'], 1.)
        self.assertEqual(cells[(0,0)]['occluded'], 1)
        self.assertEqual(cells[(0,0)]['status'], 'unknown')
        self.assertEqual(cells[(1,0)]['status'], 'unknown')


if __name__ == '__main__':
    unittest.main()
