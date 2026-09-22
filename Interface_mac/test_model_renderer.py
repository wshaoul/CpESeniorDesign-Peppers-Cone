import unittest
from dataclasses import replace

import numpy as np

from mesh_renderer import RenderSettings, checker_texture, four_view_proof, render_model
from model_warp import warp_model_views
from projection import ProjectionSettings


class ModelRendererTests(unittest.TestCase):
    def setUp(self):
        self.settings = RenderSettings(width=480,height=270)

    def test_cube_and_sphere_render_on_black(self):
        for model in ("Cube","Sphere"):
            image = render_model(replace(self.settings,model=model))
            self.assertEqual(image.shape,(270,480,3))
            self.assertFalse(image[0,0].any())
            self.assertTrue(image.any())

    def test_four_views_are_not_repeated_snapshots(self):
        views = four_view_proof(self.settings)
        self.assertEqual(len(views),4)
        for view in views[1:]:
            self.assertFalse(np.array_equal(views[0],view))
            self.assertGreater(float(np.mean(np.abs(views[0].astype(np.int16)-view.astype(np.int16)))),2)

    def test_texture_changes_cube_pixels(self):
        dark = np.full((128,128,3),25,np.uint8)
        bright = np.full((128,128,3),230,np.uint8)
        first = render_model(self.settings,dark)
        second = render_model(self.settings,bright)
        self.assertGreater(float(second.mean()),float(first.mean()))

    def test_one_and_four_view_warps(self):
        views = four_view_proof(self.settings,checker_texture(128))
        single = warp_model_views(views[:1],640,480)
        multiple = warp_model_views(views,640,480)
        self.assertEqual(single.shape,(480,640,3))
        self.assertTrue(single.any())
        self.assertTrue(multiple.any())
        self.assertFalse(np.array_equal(single,multiple))

    def test_four_warp_sectors_keep_their_own_view(self):
        colors = ((30,60,200),(40,190,70),(210,80,50),(180,60,190))
        views = [np.full((180,320,3),color,np.uint8) for color in colors]
        output = warp_model_views(views,640,640,ProjectionSettings(gain=1))
        # Top, right, bottom and left follow the default 270-degree rotation.
        for point,color in zip(((320,100),(540,320),(320,540),(100,320)),colors):
            x,y = point
            np.testing.assert_allclose(output[y,x],color,atol=1)


if __name__ == "__main__":
    unittest.main()
