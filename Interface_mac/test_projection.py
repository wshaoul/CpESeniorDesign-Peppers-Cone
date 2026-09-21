import unittest
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from projection import ProjectionSettings, LiveProjector, build_maps, alignment_pattern, test_card


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.settings = ProjectionSettings(width=640,height=480,remove_background=False)

    def test_black_stays_zero_in_every_layout(self):
        with patch("projection.make_segmenter",return_value=None):
            processor = LiveProjector()
        for views in (1,4,6,8):
            result = processor.process(np.zeros((480,640,3),np.uint8),replace(self.settings,views=views))
            self.assertEqual(result.shape,(480,640,3))
            self.assertFalse(result.any())
        processor.close()

    def test_four_rotational_copies(self):
        s = replace(self.settings,width=640,height=640,gap=0,mirror=False)
        mx,my = build_maps((720,1280,3),s)
        for x,y in ((320,90),(550,320),(320,550),(90,320)):
            self.assertAlmostEqual(float(mx[y,x]),639.5,places=2)
            self.assertAlmostEqual(float(my[y,x]),float(my[90,320]),places=2)

    def test_source_aspect_not_squashed(self):
        mx,my = build_maps((720,1280,3),replace(self.settings,mirror=False,portrait_crop=False))
        # Mapping changes source x and y on the same virtual-square scale.
        self.assertEqual(mx.dtype,np.float32)
        self.assertEqual(my.dtype,np.float32)
        self.assertGreater(float(mx.max()),1000)
        self.assertGreater(float(my.max()),720)

    def test_mirror_and_head_inversion(self):
        base = replace(self.settings,mirror=False)
        mx,my = build_maps((720,1280,3),base)
        reflected,_ = build_maps((720,1280,3),replace(base,mirror=True))
        _,inverted = build_maps((720,1280,3),replace(base,invert=True))
        valid = mx!=-1
        np.testing.assert_allclose((mx+reflected)[valid],1279,atol=.001)
        np.testing.assert_allclose((my+inverted)[valid],719,atol=.001)

    def test_outside_annulus_black_and_cache_reused(self):
        with patch("projection.make_segmenter",return_value=None):
            processor = LiveProjector()
        source = np.full((480,480,3),200,np.uint8)
        output = processor.process(source,self.settings)
        maps = processor.maps
        self.assertFalse(output[0,0].any())
        self.assertFalse(output[240,320].any())
        self.assertTrue(output.any())
        processor.process(source,self.settings)
        self.assertIs(processor.maps,maps)

    def test_all_sectors_populated(self):
        source = np.full((720,720,3),255,np.uint8)
        for count in (4,6,8):
            s = replace(self.settings,views=count)
            mx,my = build_maps(source.shape,s)
            image = cv2.remap(source,mx,my,cv2.INTER_LINEAR)
            for i in range(count):
                angle = np.deg2rad(s.rotation+i*360/count)
                x,y = round(320+150*np.cos(angle)),round(240+150*np.sin(angle))
                self.assertTrue(image[y,x].any())

    def test_bounds_and_cards(self):
        with self.assertRaises(ValueError):
            build_maps((720,1280),replace(self.settings,width=8000))
        self.assertEqual(alignment_pattern(self.settings).shape,(480,640,3))
        self.assertTrue(test_card().any())

    def test_segmentation_black_and_unity_alpha(self):
        source = np.full((720,1280,3),100,np.uint8)
        class Segmenter:
            value = 0
            def process(self,image):
                return SimpleNamespace(segmentation_mask=np.full(image.shape[:2],self.value,np.float32))
            def close(self):
                pass
        fake = Segmenter()
        with patch("projection.make_segmenter",return_value=fake):
            processor = LiveProjector()
        s = replace(self.settings,remove_background=True)
        self.assertFalse(processor.process(source,s).any())
        fake.value = 1
        segmented = processor.process(source,s)
        unsegmented = processor.process(source,s,segment=False)
        # Out-of-bounds interpolation can attenuate the very outer source edge.
        mx,my = processor.maps
        stable = (mx>10) & (mx<1269) & (my>10) & (my<709)
        self.assertTrue(stable.any())
        np.testing.assert_array_equal(segmented[stable],unsegmented[stable])
        self.assertFalse(segmented[0,0].any())
        processor.close()


if __name__ == "__main__":
    unittest.main()
