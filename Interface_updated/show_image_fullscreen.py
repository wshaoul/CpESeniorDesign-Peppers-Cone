# show_image_fullscreen.py
#
# Display a single image borderless/fullscreen on a black background, for
# physically testing a pre-warped PNG (e.g. tiger_cone_warp.png) under the
# cone reflector. Mirrors the fullscreen window setup already used by
# live_view.py's LiveView._start_fullscreen, so there's no browser/photo-app
# chrome to break the illusion.
#
# Usage:
#   winvenv310\Scripts\python.exe show_image_fullscreen.py tiger_cone_warp.png

import sys
import tkinter as tk
from PIL import Image, ImageTk


def main(path):
    root = tk.Tk()
    root.title("Cone Display")
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("q", lambda e: root.destroy())
    root.bind("Q", lambda e: root.destroy())

    label = tk.Label(root, bg="black")
    label.pack(fill="both", expand=True)

    src = Image.open(path).convert("RGB")

    def render(_=None):
        sw, sh = root.winfo_width(), root.winfo_height()
        if sw <= 1 or sh <= 1:
            return
        fw, fh = src.size
        scale = min(sw / fw, sh / fh)
        new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
        resized = src.resize((new_w, new_h), Image.LANCZOS)
        img = ImageTk.PhotoImage(resized)
        label.config(image=img)
        label.image = img  # keep a reference

    root.bind("<Configure>", render)
    root.after(50, render)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_image_fullscreen.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
