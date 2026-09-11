# studio_main.py
import tkinter as tk
from tkinter import ttk
import threading
from functools import wraps

# ---------- tiny threading helper ----------
def run_in_thread(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        t.start()
        return t
    return wrapper

# --- Import your separate pages (fallback to placeholders if not present) ---
try:
    from upload_view import UploadView
except Exception:
    class UploadView(ttk.Frame):
        def __init__(self, parent, controller=None):
            super().__init__(parent)
            ttk.Label(self, text="Upload View (placeholder)", font=("Segoe UI", 16, "bold")).pack(pady=40)

        # non-blocking stub so calls are safe
        def start_async(self):
            @run_in_thread
            def _noop(): pass
            return _noop

try:
    from record_view import RecordView
except Exception:
    class RecordView(ttk.Frame):
        def __init__(self, parent, controller=None):
            super().__init__(parent)
            ttk.Label(self, text="Record View (placeholder)", font=("Segoe UI", 16, "bold")).pack(pady=40)

        def start_async(self):
            @run_in_thread
            def _noop(): pass
            return _noop

try:
    from live_view import LiveView
except Exception:
    class LiveView(ttk.Frame):
        def __init__(self, parent, controller=None):
            super().__init__(parent)
            ttk.Label(self, text="Live View (placeholder)", font=("Segoe UI", 16, "bold")).pack(pady=40)

        def start_async(self):
            @run_in_thread
            def _noop(): pass
            return _noop

# Modern theming (falls back to ttk if ttkbootstrap isn't installed)
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
    TTKB = True
except Exception:
    TTKB = False

APP_TITLE = "Live Studio"
APP_W, APP_H = 1220, 760
LIGHT_MODE_SECONDARY = "#1f4e79"  # dark blue
DARK_MODE_SECONDARY = "#d9eaf7"   # light blue-white

# ---------- Card / Tile helpers ----------
def _create_round_rect(canvas, x1, y1, x2, y2, r=16, **kwargs):
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


def make_card(parent, title=None, subtitle=None, pad=(20, 16), rounded=18):
    """Rounded card that supports both light and dark mode."""
    wrapper = (tb.Frame if TTKB else ttk.Frame)(parent)

    cnv = tk.Canvas(
        wrapper,
        bd=0,
        highlightthickness=0
    )
    cnv.pack(fill="both", expand=True)

    content = (tb.Frame if TTKB else ttk.Frame)(
        cnv,
        padding=pad if TTKB else 0
    )

    state = {"win_id": None}

    def draw(_=None):
        dark = (
            getattr(cnv.winfo_toplevel(), "theme_mode", "light")
            == "dark"
        )

        page_bg = "#22262b" if dark else "#f8fafc"
        card_bg = "#2b3035" if dark else "#ffffff"
        border = "#495057" if dark else "#e5e7eb"
        shadow = "#181b1f" if dark else "#edf2ff"

        cnv.configure(bg=page_bg)

        cnv.delete("bg")

        w = max(cnv.winfo_width(), 10)
        h = max(cnv.winfo_height(), 10)

        _create_round_rect(
            cnv,
            10, 12, w - 6, h - 6,
            rounded + 2,
            fill=shadow,
            outline=shadow,
            tags="bg"
        )

        _create_round_rect(
            cnv,
            6, 6, w - 10, h - 10,
            rounded,
            fill=card_bg,
            outline=border,
            tags="bg"
        )

        if state["win_id"] is None:
            state["win_id"] = cnv.create_window(
                22,
                20,
                anchor="nw",
                window=content
            )

        cnv.itemconfigure(
            state["win_id"],
            width=max(1, w - 44)
        )

    cnv.bind("<Configure>", draw)

    # Allows the card to be redrawn after changing themes.
    cnv.refresh_theme = draw

    draw()

    if title:
        head = (tb.Frame if TTKB else ttk.Frame)(content)
        head.pack(fill="x", pady=(0, 8))

        (tb.Label if TTKB else ttk.Label)(
            head,
            text=title,
            font=("Segoe UI", 12, "bold")
        ).pack(side="left")

        if subtitle:
            (tb.Label if TTKB else ttk.Label)(
                head,
                text=subtitle,
                style="Muted.TLabel"
            ).pack(side="right")

        ttk.Separator(content).pack(fill="x", pady=(4, 10))

    return wrapper, content


# ---------- ActionTile (instant-click) ----------
class ActionTile((tb.Frame if TTKB else ttk.Frame)):
    """One tile with icon, title, subtitle, consistent height and instant click."""
    def __init__(self, parent, title, desc, emoji="🔴", command=None):
        super().__init__(parent)

        # inner rounded card (more padding for vertical balance)
        self.card, inner = make_card(self, pad=(18, 20), rounded=16)
        self.card.pack(fill="both", expand=True, padx=4, pady=4)
        self.card.configure(cursor="hand2", width=280, height=170)
        self.card.pack_propagate(False)

        # icon + text row centered vertically
        row = (tb.Frame if TTKB else ttk.Frame)(inner)
        row.pack(fill="both", expand=True)
        icon_lbl = (tb.Label if TTKB else ttk.Label)(row, text=emoji, font=("Segoe UI Emoji", 30))
        icon_lbl.pack(side="left", padx=(2, 16))
        textcol = (tb.Frame if TTKB else ttk.Frame)(row)
        textcol.pack(side="left", fill="x", expand=True)

        title_lbl = (tb.Label if TTKB else ttk.Label)(
            textcol, text=title, font=("Segoe UI", 12, "bold")
        )
        title_lbl.pack(anchor="w")
        desc_lbl = (tb.Label if TTKB else ttk.Label)(
            textcol, text=desc, style="Muted.TLabel"
        )
        desc_lbl.pack(anchor="w", pady=(2, 0))

        # ---- INSTANT CLICK: fire on press, bind on the outer container, stop propagation
        def on_press(_evt=None):
            if callable(command):
                command()
            return "break"  # stop other handlers to avoid delays/bubbling

        # Bind only the outer 'card' for primary behavior
        self.card.bind("<ButtonPress-1>", on_press)
        self.card.bind("<Button-1>", on_press)          # safety for some platforms
        self.card.bind("<Return>", on_press)            # keyboard activation
        self.card.bind("<space>", on_press)

        # Funnel child presses to the parent card and stop duplicate release handlers
        for w in (row, icon_lbl, textcol, title_lbl, desc_lbl):
            w.bind("<ButtonPress-1>", lambda e: on_press(e), add="+")
            w.bind("<Button-1>", lambda e: "break", add="+")

        # Cursor feedback
        def _hand(_e=None): self.card.configure(cursor="hand2")
        def _arrow(_e=None): self.card.configure(cursor="")
        for w in (self.card, row, icon_lbl, textcol, title_lbl, desc_lbl):
            w.bind("<Enter>", _hand, add="+")
            w.bind("<Leave>", _arrow, add="+")


class TileGrid((tb.Frame if TTKB else ttk.Frame)):
    """
    Grid that can be fixed to a number of columns.
    Tiles must be created as children of this frame.
    """
    def __init__(self, parent, gap_x=12, gap_y=12, fixed_cols=None):
        super().__init__(parent)
        self.tiles = []
        self.gx, self.gy = gap_x, gap_y
        self.fixed_cols = fixed_cols  # if set, always use this many columns
        self.bind("<Configure>", self._layout)

    def set_tiles(self, tiles):
        for t in tiles:
            if t.master is not self:
                raise ValueError("Each ActionTile must be created with parent=TileGrid")
        self.tiles = tiles
        self._layout()

    def _compute_cols(self, width):
        if self.fixed_cols:
            return self.fixed_cols
        # fallback responsive behavior
        if width >= 950:
            return 3
        elif width >= 640:
            return 2
        return 1

    def _layout(self, _=None):
        if not self.tiles:
            return
        w = max(self.winfo_width(), 1)
        cols = self._compute_cols(w)

        # clear and re-grid
        for child in self.tiles:
            child.grid_forget()
        for i, child in enumerate(self.tiles):
            r, c = divmod(i, cols)
            child.grid(row=r, column=c, sticky="nsew",
                       padx=(0 if c == 0 else self.gx, 0),
                       pady=(0 if r == 0 else self.gy, 0))
        # equal columns
        for c in range(cols):
            self.columnconfigure(c, weight=1, uniform="tiles")
        for c in range(cols, 6):
            self.columnconfigure(c, weight=0)


# ---------- Gradient Preview ----------
class PreviewCanvas(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, bd=0, highlightthickness=0, relief="flat", height=400)
        self._overlay = None
        self._status_l = None
        self._status_r = None
        self.bind("<Configure>", self._redraw)
        self._build_overlay()

    def _redraw(self, _=None):
        self.delete("grad")
        w, h = self.winfo_width(), self.winfo_height()
        # maintain ~16:9
        target_h = int(max(w, 1) * 9 / 16)
        if h != target_h:
            h = target_h
            self.config(height=h)

        steps = 40
        c1 = (38, 50, 77)   # slate-ish
        c2 = (16, 23, 42)
        for i in range(steps):
            t = i / (steps - 1)
            r = int(c1[0] + (c2[0]-c1[0])*t)
            g = int(c1[1] + (c2[1]-c1[1])*t)
            b = int(c1[2] + (c2[2]-c1[2])*t)
            col = f"#{r:02x}{g:02x}{b:02x}"
            y1 = int(h * i / steps); y2 = int(h * (i+1) / steps)
            self.create_rectangle(0, y1, w, y2, outline=col, fill=col, tags="grad")
        self.create_oval(int(-0.15*w), int(-1.1*h), int(0.9*w), int(h*0.40),
                         outline="", fill="#2b3e74", stipple="gray25", tags="grad")

        if self._overlay:
            self.coords(self._overlay, w//2, int(h*0.84))
            self.coords(self._status_l, 14, 14)
            self.coords(self._status_r, w-14, 14)

    def _pill_btn(self, parent, text, primary=False):
        btn = (tb.Button if TTKB else ttk.Button)(
            parent, text=text,
            **({"bootstyle": ("danger" if primary else "light-outline")} if TTKB else {})
        )
        try:
            btn.configure(width=10)
        except Exception:
            pass
        btn.pack(side="left", padx=6, pady=6, ipady=2)
        return btn

    def _build_overlay(self):
        left = (tb.Frame if TTKB else ttk.Frame)(self)
        dot = tk.Canvas(left, width=8, height=8, bd=0, highlightthickness=0)
        dot.create_oval(0, 0, 8, 8, fill="#34d399", outline="#34d399")
        dot.pack(side="left", padx=(0, 6))
        (tb.Label if TTKB else ttk.Label)(
            left, text="Ready", **({"bootstyle": "inverse-dark"} if TTKB else {})
        ).pack(side="left")

        right = (tb.Label if TTKB else ttk.Label)(
            self, text="00:00:00", **({"bootstyle": "inverse-dark"} if TTKB else {})
        )

        bar_bg = (tb.Frame if TTKB else ttk.Frame)(self)
        inner = (tb.Frame if TTKB else ttk.Frame)(bar_bg); inner.pack()
        self._pill_btn(inner, "Mic")
        self._pill_btn(inner, "Cam")
        self._pill_btn(inner, "Screen")
        spacer = (tb.Frame if TTKB else ttk.Frame)(inner); spacer.pack(side="left", padx=10)
        self._pill_btn(inner, "Start", primary=True)

        self._overlay = self.create_window(0, 0, window=bar_bg, anchor="center")
        self._status_l = self.create_window(0, 0, window=left, anchor="nw")
        self._status_r = self.create_window(0, 0, window=right, anchor="ne")


# --------------------- Pages ---------------------
class HomePage(ttk.Frame if not TTKB else tb.Frame):
    """Dashboard page that contains the header, quick actions, preview, devices, footer."""
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Root area
        root = (tb.Frame if TTKB else ttk.Frame)(self, padding=(8, 8))
        root.pack(fill="both", expand=True)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        # Header
        header = (tb.Frame if TTKB else ttk.Frame)(root, padding=(16, 10))
        header.grid(row=0, column=0, sticky="ew")
        leftgrp = (tb.Frame if TTKB else ttk.Frame)(header); leftgrp.pack(side="left")
        icon = (tb.Frame if TTKB else ttk.Frame)(leftgrp, width=28, height=28)
        if TTKB: icon.configure(bootstyle="primary")
        icon.pack_propagate(False); icon.pack(side="left", padx=(0, 10))
        (tb.Label if TTKB else ttk.Label)(leftgrp, text="Live Studio", font=("Segoe UI", 18, "bold")).pack(side="left")

        rightgrp = (tb.Frame if TTKB else ttk.Frame)(header)
        rightgrp.pack(side="right")
        self.settings_button = (tb.Button if TTKB else ttk.Button)(
            rightgrp,
            text="Settings",
            command=self.controller.open_settings,
            **({"bootstyle": "primary-outline"} if TTKB else {})
        )
        self.settings_button.pack(side="left")

        # Body grid
        # Centered dashboard body
        body = (tb.Frame if TTKB else ttk.Frame)(
            root,
            padding=12
        )
        body.grid(row=1, column=0, sticky="nsew")

        # Empty outside columns center the content column.
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        content = (tb.Frame if TTKB else ttk.Frame)(body)
        content.grid(row=0, column=0)
        content.columnconfigure(0, weight=1)


        # ---------- Quick Actions ----------
        qa_card, qa = make_card(
            content,
            "Quick Actions",
            "Choose what you want to do."
        )
        qa_card.grid(row=0, column=0, sticky="ew")

        # Control the card height.
        qa_card.configure(width=1100, height=310)
        qa_card.pack_propagate(False)

        tile_grid = TileGrid(
            qa,
            gap_x=12,
            gap_y=12,
            fixed_cols=3
        )
        tile_grid.pack(fill="x")

        tiles = [
            ActionTile(
                tile_grid,
                "Go Live",
                "Broadcast to RTMP",
                "🔴",
                command=lambda: self._nav_and_boot("LiveView")
            ),
            ActionTile(
                tile_grid,
                "Record",
                "Save locally",
                "🎬",
                command=lambda: self._nav_and_boot("RecordView")
            ),
            ActionTile(
                tile_grid,
                "Upload",
                "Send a file",
                "📤",
                command=lambda: self._nav_and_boot("UploadView")
            ),
        ]

        tile_grid.set_tiles(tiles)

        # ---------- Recent ----------
        recent_card, recent = make_card(
            content,
            "Recent"
        )
        recent_card.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(12, 0)
        )

        recent_card.configure(width=1100,height=180)
        recent_card.pack_propagate(False)

        rgrid = (tb.Frame if TTKB else ttk.Frame)(recent)
        rgrid.pack(fill="x")

        (tb.Label if TTKB else ttk.Label)(
            rgrid,
            text="Recent recordings",
            font=("", 10, "bold")
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        (tb.Label if TTKB else ttk.Label)(
            rgrid,
            text="Uploads",
            font=("", 10, "bold")
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(16, 0)
        )

        (tb.Label if TTKB else ttk.Label)(
            rgrid,
            text="hi.mp4 — 2 min ago",
            style="Muted.TLabel"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(4, 0)
        )

        (tb.Label if TTKB else ttk.Label)(
            rgrid,
            text="demo.mov — yesterday",
            style="Muted.TLabel"
        ).grid(
            row=1,
            column=1,
            sticky="w",
            padx=(16, 0),
            pady=(4, 0)
        )

        rgrid.columnconfigure((0, 1), weight=1, uniform="rcols")

        # ---------- Devices ----------
        dev_card, dev = make_card(
            content,
            "Devices"
        )
        dev_card.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(12, 0)
        )

        dev_card.configure(width=1100, height=220)
        dev_card.pack_propagate(False)

        device_grid = (tb.Frame if TTKB else ttk.Frame)(dev)
        device_grid.pack(fill="x")

        devices = [
            ("Camera", "Integrated (1080p)"),
            ("Microphone", "Yeti Nano"),
            ("Speakers", "Realtek"),
        ]

        for i, (device_name, device_value) in enumerate(devices):
            (tb.Label if TTKB else ttk.Label)(
                device_grid,
                text=device_name,
                style="Muted.TLabel"
            ).grid(
                row=i,
                column=0,
                sticky="w",
                pady=4
            )

            (tb.Label if TTKB else ttk.Label)(
                device_grid,
                text=device_value
            ).grid(
                row=i,
                column=1,
                sticky="e",
                pady=4
            )

        device_grid.columnconfigure(1, weight=1)

        # Footer
        footer = (tb.Frame if TTKB else ttk.Frame)(root, padding=(16, 10))
        footer.grid(row=2, column=0, sticky="ew")
        (tb.Label if TTKB else ttk.Label)(footer, text="● Ready",
                                          **({"bootstyle": "success"} if TTKB else {})).pack(side="left")
        rightbtns = (tb.Frame if TTKB else ttk.Frame)(footer)
        rightbtns.pack(side="right")

        self.quit_button = (tb.Button if TTKB else ttk.Button)(
            rightbtns,
            text="Quit",
            command=self.controller.destroy,
            **({"bootstyle": "danger-outline"} if TTKB else {})
        )
        self.quit_button.pack(side="left", padx=4)

    def _nav_and_boot(self, page_name: str):
        """Raise page immediately; start heavy work on a background thread (if defined)."""
        self.controller.show_page(page_name)        # instant UI response
        page = self.controller.pages.get(page_name)
        if hasattr(page, "start_async"):
            starter = page.start_async()            # returns a thread-wrapped callable
            if callable(starter):
                starter()                           # run heavy init off the UI thread


# ---------- App (Router) ----------
def refresh_custom_theme(widget):
    """Redraw custom Canvas cards after changing themes."""
    refresh = getattr(widget, "refresh_theme", None)

    if callable(refresh):
        refresh()

    for child in widget.winfo_children():
        refresh_custom_theme(child)


def open_settings_dialog(app):
    """Open the settings window with a dark-mode option."""
    existing = getattr(app, "settings_window", None)

    try:
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
    except tk.TclError:
        pass

    window_cls = tb.Toplevel if TTKB else tk.Toplevel
    win = window_cls(app)

    app.settings_window = win

    win.title("Settings")
    win.geometry("380x190")
    win.resizable(False, False)
    win.transient(app)

    body_cls = tb.Frame if TTKB else ttk.Frame
    label_cls = tb.Label if TTKB else ttk.Label
    check_cls = tb.Checkbutton if TTKB else ttk.Checkbutton
    button_cls = tb.Button if TTKB else ttk.Button

    body = body_cls(win, padding=24)
    body.pack(fill="both", expand=True)

    label_cls(
        body,
        text="Appearance",
        font=("Segoe UI", 14, "bold")
    ).pack(anchor="w")

    label_cls(
        body,
        text="Choose how the Live Studio interface is displayed."
    ).pack(anchor="w", pady=(4, 18))

    dark_var = tk.BooleanVar(
        value=app.theme_mode == "dark"
    )

    toggle_options = (
        {"bootstyle": "success-round-toggle"}
        if TTKB
        else {}
    )

    check_cls(
        body,
        text="Dark mode",
        variable=dark_var,
        command=lambda: app.set_dark_mode(dark_var.get()),
        **toggle_options
    ).pack(side="left")

    button_cls(
        body,
        text="Close",
        command=win.destroy
    ).pack(side="right")

    win.protocol("WM_DELETE_WINDOW", win.destroy)
    win.grab_set()

if TTKB:
    class App(tb.Window):
        def __init__(self):
            super().__init__(themename="flatly")
            self.theme_mode = "light"
            self.settings_window = None
            self.style.configure(
                "Muted.TLabel",
                foreground=LIGHT_MODE_SECONDARY
            )
            self.title(APP_TITLE)
            self.geometry(f"{APP_W}x{APP_H}")
            self.minsize(1100, 680)

            # container for pages
            container = tb.Frame(self)
            container.pack(fill="both", expand=True)
            container.grid_rowconfigure(0, weight=1)
            container.grid_columnconfigure(0, weight=1)

            self.pages = {}
            for Page in (HomePage, LiveView, UploadView, RecordView):
                page = Page(container, self)
                self.pages[Page.__name__] = page
                page.grid(row=0, column=0, sticky="nsew")

            self.show_page("HomePage")
            self.state('zoomed')  # Start maximized on Windows

        def show_page(self, name: str):
            self.pages[name].tkraise()

        def open_settings(self):
            open_settings_dialog(self)

        def set_dark_mode(self, enabled: bool):
            self.theme_mode = "dark" if enabled else "light"

            self.style.theme_use(
                "darkly" if enabled else "flatly"
            )

            secondary_color = (
                DARK_MODE_SECONDARY
                if enabled
                else LIGHT_MODE_SECONDARY
            )

            self.style.configure(
                "Muted.TLabel",
                foreground=secondary_color
            )

            home = self.pages.get("HomePage")

            if home is not None:
                button_style = (
                    "light-outline"
                    if enabled
                    else "primary-outline"
                )

                home.settings_button.configure(
                    bootstyle=button_style
                )

                home.quit_button.configure(
                    bootstyle=button_style
                )

            refresh_custom_theme(self)

else:
    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.theme_mode = "light"
            self.settings_window = None
            self.style = ttk.Style(self)
            self.style.configure(
                "Muted.TLabel",
                foreground=LIGHT_MODE_SECONDARY
            )
            self.title(APP_TITLE)
            self.geometry(f"{APP_W}x{APP_H}")
            container = ttk.Frame(self)
            container.pack(fill="both", expand=True)
            container.grid_rowconfigure(0, weight=1)
            container.grid_columnconfigure(0, weight=1)

            self.pages = {}
            for Page in (HomePage, LiveView, UploadView, RecordView):
                page = Page(container, self)
                self.pages[Page.__name__] = page
                page.grid(row=0, column=0, sticky="nsew")

            self.show_page("HomePage")

        def show_page(self, name: str):
            self.pages[name].tkraise()

        def open_settings(self):
            open_settings_dialog(self)

        def set_dark_mode(self, enabled: bool):
            self.theme_mode = "dark" if enabled else "light"
            bg = "#22262b" if enabled else "#f8fafc"
            fg = "#f8f9fa" if enabled else "#212529"
            self.configure(bg=bg)
            self.style.configure(".", background=bg, foreground=fg)
            self.style.configure("TFrame", background=bg)
            self.style.configure("TLabel", background=bg, foreground=fg)
            self.style.configure(
                "Muted.TLabel",
                background=bg,
                foreground=(
                    DARK_MODE_SECONDARY
                    if enabled
                    else LIGHT_MODE_SECONDARY
                )
            )
            refresh_custom_theme(self)


if __name__ == "__main__":
    app = App()
    app.mainloop()
