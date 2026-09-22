"""Small CPU 3D renderer used to validate geometry before live reconstruction."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RenderSettings:
    width: int = 960
    height: int = 540
    yaw: float = 25
    elevation: float = 15
    fov: float = 42
    scale: float = 1.0
    model: str = "Cube"


def checker_texture(size=512):
    yy, xx = np.indices((size, size))
    cells = ((xx // (size // 8)) + (yy // (size // 8))) % 2
    image = np.zeros((size, size, 3), np.uint8)
    image[cells == 0] = (235, 115, 35)
    image[cells == 1] = (40, 205, 245)
    cv2.putText(image, "3D", (size//3, size//2+35), cv2.FONT_HERSHEY_SIMPLEX,
                2.5, (255, 255, 255), 10, cv2.LINE_AA)
    return image


def _rotation(yaw, elevation):
    y, x = np.deg2rad([yaw, elevation])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]], np.float32)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]], np.float32)
    return rx @ ry


def _project(vertices, settings):
    points = vertices @ _rotation(settings.yaw, settings.elevation).T
    points *= settings.scale
    z = points[:, 2] + 4.2
    focal = .5 * settings.height / np.tan(np.deg2rad(settings.fov) / 2)
    screen = np.column_stack((settings.width/2 + focal*points[:, 0]/z,
                              settings.height/2 - focal*points[:, 1]/z))
    return points, screen.astype(np.float32), z


def _cube():
    vertices = np.float32([
        [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
        [-1,-1, 1], [1,-1, 1], [1,1, 1], [-1,1, 1],
    ])
    faces = [(0,1,2,3), (5,4,7,6), (4,0,3,7),
             (1,5,6,2), (3,2,6,7), (4,5,1,0)]
    return vertices, faces


def _render_cube(settings, texture):
    vertices, faces = _cube()
    rotated, points, depth = _project(vertices, settings)
    canvas = np.zeros((settings.height, settings.width, 3), np.uint8)
    source = np.float32([[0,0], [texture.shape[1]-1,0],
                         [texture.shape[1]-1,texture.shape[0]-1],
                         [0,texture.shape[0]-1]])
    light = np.array([-.3, -.5, -1], np.float32)
    light /= np.linalg.norm(light)
    ordered = sorted(enumerate(faces), key=lambda item: float(np.mean(depth[list(item[1])])), reverse=True)
    names = ("FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM")
    colors = ((40,90,220), (210,90,45), (65,190,90), (190,75,190),
              (45,180,220), (180,150,60))
    for face_index, face in ordered:
        polygon = points[list(face)]
        a, b, c = rotated[list(face)[:3]]
        normal = np.cross(b-a, c-a)
        normal /= max(1e-6, np.linalg.norm(normal))
        center_from_camera = np.mean(rotated[list(face)],axis=0) + np.array([0,0,4.2])
        if np.dot(normal,center_from_camera) >= 0:
            continue
        shade = .42 + .58*abs(float(np.dot(normal, light)))
        face_texture = cv2.addWeighted(texture,.78,
            np.full_like(texture,colors[face_index]),.22,0)
        font_scale = max(.5,min(face_texture.shape[:2])/260)
        cv2.putText(face_texture,names[face_index],
                    (round(face_texture.shape[1]*.08),round(face_texture.shape[0]*.9)),
                    cv2.FONT_HERSHEY_SIMPLEX,font_scale,(255,255,255),
                    max(1,round(font_scale*2)),cv2.LINE_AA)
        matrix = cv2.getPerspectiveTransform(source, polygon)
        warped = cv2.warpPerspective(face_texture, matrix, (settings.width, settings.height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
        mask = np.zeros((settings.height, settings.width), np.uint8)
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255, cv2.LINE_AA)
        shaded = np.clip(warped.astype(np.float32)*shade, 0, 255).astype(np.uint8)
        canvas[mask > 0] = shaded[mask > 0]
        cv2.polylines(canvas, [np.round(polygon).astype(np.int32)], True,
                      (245,245,245), 2, cv2.LINE_AA)
    return canvas


def _sphere_mesh(rows=16, columns=32):
    vertices, uv = [], []
    for row in range(rows+1):
        v = row/rows
        phi = np.pi*v
        for column in range(columns):
            u = column/columns
            theta = 2*np.pi*u
            vertices.append((np.sin(phi)*np.sin(theta), np.cos(phi),
                             np.sin(phi)*np.cos(theta)))
            uv.append((u, v))
    faces = []
    for row in range(rows):
        for column in range(columns):
            nxt = (column+1)%columns
            a, b = row*columns+column, row*columns+nxt
            c, d = (row+1)*columns+nxt, (row+1)*columns+column
            faces.extend(((a,b,c), (a,c,d)))
    return np.float32(vertices), np.float32(uv), faces


def _render_sphere(settings, texture):
    vertices, uv, faces = _sphere_mesh()
    rotated, points, depth = _project(vertices, settings)
    canvas = np.zeros((settings.height, settings.width, 3), np.uint8)
    th, tw = texture.shape[:2]
    light = np.array([-.4, -.6, -1], np.float32)
    light /= np.linalg.norm(light)
    ordered = sorted(faces, key=lambda face: float(np.mean(depth[list(face)])), reverse=True)
    for face in ordered:
        ids = list(face)
        polygon = points[ids]
        a, b, c = rotated[ids]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-6:
            continue
        normal /= length
        center_from_camera = np.mean(rotated[ids],axis=0) + np.array([0,0,4.2])
        if np.dot(normal,center_from_camera) >= 0:
            continue
        shade = .35 + .65*abs(float(np.dot(normal, light)))
        u, v = np.mean(uv[ids], axis=0)
        color = texture[min(th-1, round(v*(th-1))), min(tw-1, round(u*(tw-1)))]
        color = tuple(int(x) for x in np.clip(color*shade, 0, 255))
        cv2.fillConvexPoly(canvas, np.round(polygon).astype(np.int32), color,
                           cv2.LINE_AA)
    return canvas


def render_model(settings, texture=None):
    if settings.width < 160 or settings.height < 120:
        raise ValueError("Render size is too small")
    texture = checker_texture() if texture is None else texture
    if settings.model == "Cube":
        return _render_cube(settings, texture)
    if settings.model == "Sphere":
        return _render_sphere(settings, texture)
    raise ValueError(f"Unknown model: {settings.model}")


def four_view_proof(settings, texture=None):
    """Render genuinely different front/right/back/left virtual viewpoints."""
    views = []
    for offset in (0, 90, 180, 270):
        current = RenderSettings(**{**settings.__dict__, "yaw": settings.yaw+offset})
        views.append(render_model(current, texture))
    return views
