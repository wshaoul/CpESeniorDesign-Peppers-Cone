"""Uncalibrated layouts for testing 3D views on the physical cone."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np

from projection import ProjectionSettings, build_maps


@lru_cache(maxsize=24)
def _sector_maps(source_shape, width, height, base, index, count):
    s = replace(base, width=width, height=height, views=count,
                rotation=base.rotation + index*(360/count if count > 1 else 0))
    maps = build_maps(source_shape, s)
    if count > 1:
        yy,xx = np.ogrid[:height,:width]
        dx = xx-s.center_x*width
        dy = yy-s.center_y*height
        angle = (np.arctan2(dy,dx)-np.deg2rad(s.rotation)+np.pi)%(2*np.pi)-np.pi
        sector = np.abs(angle)<=np.pi/count
        maps = tuple(np.where(sector,m,-1).astype(np.float32) for m in maps)
    return maps


def warp_model_views(views, width, height, settings=None):
    """Place one or four distinct rendered viewpoints into a cone layout.

    This is a starting geometric warp, not the measured distortion map required
    for perspective-correct reflection through a real cone.
    """
    count = len(views)
    if count not in (1, 4):
        raise ValueError("Model output supports one or four views")
    base = settings or ProjectionSettings()
    canvas = np.zeros((height, width, 3), np.uint8)
    for index, view in enumerate(views):
        maps = _sector_maps(view.shape,width,height,base,index,count)
        warped = cv2.remap(view, *maps, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        np.maximum(canvas, warped, out=canvas)
    if base.gain != 1:
        lut = np.clip(np.arange(256)*base.gain,0,255).astype(np.uint8)
        canvas = cv2.LUT(canvas,lut)
    return canvas
