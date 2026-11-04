"""Compact live studio layout shared by single-arc and circle renderers."""
import time
import tkinter as tk
from tkinter import ttk
from studio_theme import Card, BG, MUTED


def find_displays(view):
    """Read Windows monitor bounds without requiring another dependency."""
    displays = []
    try:
        import ctypes
        from ctypes import wintypes
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE,
                                          wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
        def collect(monitor, dc, rect, data):
            r = rect.contents
            displays.append((r.left, r.top, r.right-r.left, r.bottom-r.top))
            return True
        callback = callback_type(collect)
        ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0)
    except (AttributeError, OSError):
        pass
    return displays or [(0, 0, view.winfo_screenwidth(), view.winfo_screenheight())]


def place_output(view):
    index = max(0, view.display_choice.current())
    x, y, width, height = view._displays[index]
    # Borderless monitor-sized output also works on monitors left of the primary.
    view.fs_win.overrideredirect(True)
    view.fs_win.geometry(f"{width}x{height}{x:+d}{y:+d}")
    view.fs_win.update_idletasks()
    try:
        import ctypes
        from ctypes import wintypes
        move = ctypes.windll.user32.SetWindowPos
        move.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int, wintypes.UINT)
        move(view.fs_win.winfo_id(), 0, x, y, width, height, 0x0040)
    except (AttributeError, OSError):
        pass
    view.fs_win.lift()
    view.fs_win.focus_force()


def _preview(parent, title, caption, message):
    card = Card(parent, padding=8)
    heading = ttk.Frame(card.content)
    heading.pack(fill="x", pady=(0, 8))
    ttk.Label(heading, text=title, font=("Segoe UI", 11, "bold")).pack(side="left")
    ttk.Label(heading, text=caption, foreground="#7b8799", font=("Segoe UI", 8, "bold")).pack(side="right")
    surface = tk.Frame(card.content, bg="#10151d", width=1, height=1)
    surface.pack(fill="both", expand=True)
    surface.pack_propagate(False)
    label = tk.Label(surface, bg="#10151d", fg="#94a3b8", text=message, bd=0)
    label.place(relx=.5, rely=.5, anchor="center")
    return card, surface, label


def arrange_live(view, top, controls, left, old_card, sections, toggle, tuning, actions, status_box):
    """Keep existing controls and callbacks while replacing their presentation."""
    controls.canvas.configure(width=300)
    for section in sections:
        section.pack_forget()
    toggle.pack_forget()
    tuning.pack_forget()
    setup = ttk.Frame(left)
    setup.pack(fill="x")
    ttk.Label(setup, text="Set up your display", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 16))
    ttk.Label(setup, text="Show on", style="Body.TLabel").pack(anchor="w", pady=(0, 5))
    view.display_choice = ttk.Combobox(setup, state="readonly")
    view.display_choice.pack(fill="x", pady=(0, 12))
    def refresh_displays():
        view._displays = find_displays(view)
        view.display_choice.configure(values=[f"Display {i+1} · {w} × {h}" for i, (_, _, w, h) in enumerate(view._displays)])
        view.display_choice.current(0)
    refresh_displays()
    ttk.Label(setup, text="Picture quality", style="Body.TLabel").pack(anchor="w", pady=(0, 5))
    quality = ttk.Combobox(setup, state="readonly", values=("Smoother motion (recommended)", "Sharper picture"))
    quality.current(0)
    quality.pack(fill="x", pady=(0, 10))
    view._cone_preview_interval = .1
    def change_quality(event=None):
        view._cone_preview_interval = .1 if quality.current() == 0 else .2
        view.res_combo.set("1280x720" if quality.current() == 0 else "1920x1080")
        if view._preview_thread and view._preview_thread.is_alive():
            view._stop_preview()
            view._start_preview()
    quality.bind("<<ComboboxSelected>>", change_quality)
    view.remove_background = tk.BooleanVar(value=True)
    ttk.Checkbutton(setup, text="Remove camera background", variable=view.remove_background).pack(anchor="w", pady=(0, 12))
    ttk.Button(setup, text="Find my TV", style="TButton", command=refresh_displays).pack(anchor="w", pady=(0, 10))
    expanded = tk.BooleanVar(value=False)
    def advanced():
        if expanded.get():
            controls.canvas.configure(width=410)
            for section in (*sections, tuning):
                section.pack(fill="x", padx=4, pady=(0, 10))
        else:
            for section in (*sections, tuning):
                section.pack_forget()
            controls.canvas.configure(width=300)
        controls.canvas.yview_moveto(0)
    def toggle_advanced():
        expanded.set(not expanded.get())
        advanced()
        advanced_button.configure(text="Hide advanced settings" if expanded.get() else "Advanced settings")
    advanced_button = ttk.Button(setup, text="Advanced settings", command=toggle_advanced)
    advanced_button.pack(anchor="w")
    ttk.Label(setup, text="Tip: extend your desktop to your TV in Windows Display settings. "
              "Press Esc to close the output.", wraplength=255, style="Body.TLabel").pack(anchor="w", pady=(12, 20))
    old_card.destroy()
    previews = ttk.Frame(top, style="Shell.TFrame")
    previews.pack(side="right", fill="both", expand=True)
    previews.columnconfigure(0, weight=1)
    previews.rowconfigure((0, 1), weight=1, uniform="preview")
    camera, view._preview_container, view._preview_label = _preview(
        previews, "Your camera", "LIVE SOURCE", "Press Start camera to begin")
    camera.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
    output, view._output_container, view._output_label = _preview(
        previews, "Cone preview", "TV OUTPUT", "Your cone image will appear here")
    output.grid(row=1, column=0, sticky="nsew")
    view._last_cone_preview = 0
    actions.configure(style="Shell.TFrame")
    status_box.configure(style="Shell.TFrame")
    view.btn_preview.configure(style="Action.Primary.TButton")
    view.btn_preview.grid_configure(column=0)
    view.btn_fullscreen.configure(style="Action.TV.TButton")
    view.btn_fullscreen.grid_configure(column=1)
    view.btn_stop_preview.configure(text="Stop", style="Action.TButton", command=lambda: stop_live(view))
    view.btn_stop_preview.grid_configure(column=2)
    view.btn_close_fullscreen.grid_remove()
    view.btn_align.grid_remove()
    # Keep the less common output controls with the advanced settings.
    ttk.Button(tuning, text="Close TV output", command=view._stop_fullscreen).grid(row=20, column=0, columnspan=3, pady=6)
    ttk.Button(tuning, text="Show alignment pattern", command=view._toggle_alignment_pattern).grid(row=21, column=0, columnspan=3, pady=6)
    view.status.set("Ready when you are. Press Start camera.")


def stop_live(view):
    view._stop_fullscreen()
    view._stop_preview()
    with view._frame_lock:
        view._last_bgr = None
    for label, text in ((view._preview_label, "Press Start camera to begin"),
                        (view._output_label, "Your cone image will appear here")):
        label.configure(image="", text=text)
        label.image = None
    view.status.set("Camera stopped. Ready when you are.")


def update_cone_preview(view, frame):
    """Limit the extra warp to 5–10 fps while keeping the source preview smooth."""
    now = time.monotonic()
    if now - view._last_cone_preview < view._cone_preview_interval:
        return
    view._last_cone_preview = now
    import cv2
    from PIL import Image, ImageTk
    renderer = getattr(view, "_apply_circle_hologram", None) or view._apply_warp
    output = renderer(frame, use_segmentation=view.remove_background.get())
    height, width = output.shape[:2]
    scale = min(max(1, view._output_container.winfo_width()) / width,
                max(1, view._output_container.winfo_height()) / height)
    output = cv2.resize(output, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)))
    view._output_label.configure(image=image, text="")
    view._output_label.image = image
