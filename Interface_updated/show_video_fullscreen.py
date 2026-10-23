# show_video_fullscreen.py
#
# Loop a video borderless/fullscreen on a black background, for physically
# testing a pre-warped clip (e.g. tiger_walk.mp4) under the cone reflector.
# Same borderless/black-bg convention as show_image_fullscreen.py and
# live_view.py's own fullscreen window — no browser/player chrome to break
# the illusion.
#
# Usage:
#   winvenv310\Scripts\python.exe show_video_fullscreen.py tiger_walk.mp4

import sys
import tkinter as tk
import cv2
from PIL import Image, ImageTk


def main(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Could not open {path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    delay_ms = max(1, int(1000 / fps))

    root = tk.Tk()
    root.title("Cone Display")
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("q", lambda e: root.destroy())
    root.bind("Q", lambda e: root.destroy())

    label = tk.Label(root, bg="black")
    label.pack(fill="both", expand=True)

    def tick():
        ok, frame = cap.read()
        if not ok:  # end of clip — loop back to the start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                root.after(delay_ms, tick)
                return

        sw, sh = root.winfo_width(), root.winfo_height()
        if sw > 1 and sh > 1:
            fh, fw = frame.shape[:2]
            scale = min(sw / fw, sh / fh)
            new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.config(image=img)
        label.image = img  # keep a reference

        root.after(delay_ms, tick)

    root.after(50, tick)
    root.mainloop()
    cap.release()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_video_fullscreen.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
