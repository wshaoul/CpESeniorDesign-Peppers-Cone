"""Tk-native processed-video players for the cone and circle layouts."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from platform_support import create_selfie_segmenter


class ConeVideoPlayer(tk.Toplevel):
    """Play processed video fullscreen without using OpenCV's GUI thread."""

    def __init__(self, owner, input_path, processor, title, output_path=None,
                 resources=()):
        super().__init__(owner.winfo_toplevel())
        self._processor = processor
        self._resources = tuple(resources)
        self._closed = False
        self._writer = None
        self._output_path = output_path

        self._cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(input_path)
        if not self._cap.isOpened():
            self._cap.release()
            self.destroy()
            raise RuntimeError(f"Could not open video:\n{input_path}")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not np.isfinite(fps) or fps < 1.0:
            fps = 30.0
        self._delay_ms = max(1, round(1000.0 / min(60.0, fps)))

        if output_path:
            size = processor.output_size
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(output_path, fourcc, fps, size)
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                self._output_path = None

        self.title(title)
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda _event: self.close())
        self.bind("q", lambda _event: self.close())
        self.bind("Q", lambda _event: self.close())

        self._label = tk.Label(self, bg="black", bd=0, highlightthickness=0)
        self._label.pack(fill="both", expand=True)
        self._hint = tk.Label(
            self,
            text="Q / Esc: exit",
            fg="#aaaaaa",
            bg="#1a1a1a",
            padx=14,
            pady=5,
        )
        self._hint.place(relx=0.5, rely=1.0, anchor="s", y=-14)
        self.after(0, self._tick)

    def _tick(self):
        if self._closed:
            return
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            self.close()
            return

        try:
            processed = self._processor(frame)
        except Exception as exc:
            self.close()
            messagebox.showerror("Cone Screen", f"Could not process video:\n{exc}")
            return

        if self._writer is not None:
            self._writer.write(processed)

        sw = max(1, self.winfo_width())
        sh = max(1, self.winfo_height())
        fh, fw = processed.shape[:2]
        scale = min(sw / fw, sh / fh)
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        display = cv2.resize(
            processed,
            (max(1, int(fw * scale)), max(1, int(fh * scale))),
            interpolation=interpolation,
        )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._label.configure(image=self._image)
        self.after(self._delay_ms, self._tick)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._cap.release()
        if self._writer is not None:
            self._writer.release()
        for resource in self._resources:
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        saved_path = self._output_path
        self.destroy()
        if saved_path:
            messagebox.showinfo("Cone Screen", f"Processed copy saved:\n{saved_path}")


class _Processor:
    def __init__(self, function, output_size):
        self._function = function
        self.output_size = output_size

    def __call__(self, frame):
        return self._function(frame)


def open_single_cone_player(owner, input_path: str, save_copy: bool):
    processor, resources = create_single_processor()
    output_path = os.path.splitext(input_path)[0] + "_cone.mp4" if save_copy else None
    return ConeVideoPlayer(owner, input_path, processor,
                           "Pepper's Cone — Processed Video", output_path,
                           resources=resources)


def create_single_processor():
    """Build the single-cone frame processor independently of the GUI."""
    from live_view import (
        CANVAS_SIZE,
        FRAME_SIZE,
        build_cone_maps,
        enhance_saturation_contrast,
    )

    map_x, map_y = build_cone_maps(
        frame_size=FRAME_SIZE,
        canvas_size=CANVAS_SIZE,
        span_deg=200,
        rotate_deg=270.0,
        r_inner_frac=0.08,
        r_outer_frac=0.995,
        center_frac=(0.50, 0.50),
        radius_frac=1.00,
    )
    segmenter = create_selfie_segmenter(model_selection=1)

    def process(frame):
        square = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        foreground = _segment_person(square, segmenter)
        scaled = cv2.resize(foreground, (0, 0), fx=0.6, fy=0.6,
                            interpolation=cv2.INTER_AREA)
        padded = np.zeros_like(foreground)
        y = (FRAME_SIZE - scaled.shape[0]) // 2
        x = (FRAME_SIZE - scaled.shape[1]) // 2
        padded[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
        warped = cv2.remap(
            padded, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        return enhance_saturation_contrast(
            warped, saturation_scale=1.4, contrast_alpha=1.8,
            brightness_beta=-25,
        )

    processor = _Processor(process, (CANVAS_SIZE, CANVAS_SIZE))
    resources = (segmenter,) if segmenter else ()
    return processor, resources


def open_circle_cone_player(owner, input_path: str, save_copy: bool):
    processor, resources = create_circle_processor()
    output_path = os.path.splitext(input_path)[0] + "_circle.mp4" if save_copy else None
    return ConeVideoPlayer(owner, input_path, processor,
                           "Pepper's Cone — Circle Video", output_path,
                           resources=resources)


def create_circle_processor():
    """Build the four-view frame processor independently of the GUI."""
    from live_view_circle import (
        CANVAS_SIZE,
        FRAME_SIZE,
        apply_four_cone_views,
        build_four_cone_maps,
    )

    maps = build_four_cone_maps(
        frame_size=FRAME_SIZE,
        canvas_size=CANVAS_SIZE,
        span_deg=90,
        r_inner_frac=0.08,
        r_outer_frac=0.995,
        center_frac=(0.50, 0.50),
        radius_frac=1.00,
        base_rotate=270.0,
    )
    segmenter = create_selfie_segmenter(model_selection=1)

    def process(frame):
        square = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        foreground = _segment_person(square, segmenter)
        return apply_four_cone_views(foreground, maps, frame_size=FRAME_SIZE)

    processor = _Processor(process, (CANVAS_SIZE, CANVAS_SIZE))
    resources = (segmenter,) if segmenter else ()
    return processor, resources


def _segment_person(square, segmenter):
    if segmenter is None:
        return square
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)
    mask = cv2.GaussianBlur(result.segmentation_mask.astype("float32"), (7, 7), 0)
    mask = (mask > 0.35).astype("uint8") * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    return cv2.bitwise_and(square, square, mask=mask)
