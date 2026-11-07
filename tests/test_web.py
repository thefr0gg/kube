import unittest

import numpy as np

from kube import Kube
from kube.web.mesh import _FACE_VERTS, build_shell_mesh, build_slice_mesh
from kube.web.server import create_app


class MeshTests(unittest.TestCase):
    def test_shell_mesh_has_only_boundary_faces(self):
        kube = Kube(3, 3, 3, states=4)
        faces = kube.faces_to_numpy()
        mesh = build_shell_mesh(faces, lod=1)
        self.assertGreater(mesh["positions"].shape[0], 0)
        self.assertEqual(mesh["positions"].shape[0] % 4, 0)
        self.assertEqual(mesh["indices"].shape[0] % 6, 0)

    def test_face_vert_winding_is_outward(self):
        # Each face's first triangle (v0, v1, v2) must wind so that
        # cross(v1-v0, v2-v0) points along that face's own outward normal.
        # A flipped quad here breaks browser-side face picking (the mesh
        # looks fine with DoubleSide materials, but raycaster.face.normal
        # ends up backwards), so this guards against the winding regressing.
        expected_normals = [
            (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1),
        ]
        for face_index, expected in enumerate(expected_normals):
            v0, v1, v2, _v3 = _FACE_VERTS[face_index]
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)
            np.testing.assert_allclose(normal, expected, atol=1e-6)

    def test_slice_mesh_covers_one_plane(self):
        kube = Kube(4, 4, 4, states=4)
        faces = kube.faces_to_numpy()
        mesh = build_slice_mesh(faces, slice_axis=2, slice_index=1, lod=1)
        # one quad (4 verts) per cell in the 4x4 plane, 6 faces per cell
        self.assertEqual(mesh["positions"].shape[0], 4 * 4 * 6 * 4)

    def test_empty_kube_produces_empty_mesh(self):
        kube = Kube(1, 1, 1, states=2)
        faces = kube.faces_to_numpy()
        mesh = build_shell_mesh(faces, lod=1, visible_faces=(False,) * 6)
        self.assertEqual(mesh["positions"].shape[0], 0)
        self.assertEqual(mesh["indices"].shape[0], 0)


class ServerTests(unittest.TestCase):
    def test_create_app_builds_without_error(self):
        kube = Kube(2, 2, 2, states=4)
        app = create_app(kube)
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
