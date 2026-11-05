"""Shared Tk appearance and navigation, matching Interface_mac's studio palette."""
import importlib
import json
import os
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import ttk

BG = "#f3f5f9"
WHITE = "#ffffff"
INK = "#192336"
MUTED = "#66748b"
BLUE = "#315ce8"
NAVY = "#17243e"


# Map the existing studio palette so both renderers share the same appearance.
DARK_COLORS = {
    BG: "#111827", WHITE: "#1f2937", INK: "#e5e7eb", MUTED: "#a8b5c9",
    NAVY: "#dbe7ff", "#15213a": "#e5edff", "#7b8799": "#a8b5c9",
    "#dce3ef": "#455268", "#e0e5ef": "#455268", "#f8f9fc": "#162032",
    "#f0f4fd": "#2c3b53", "#e3ebfc": "#354762", "#edf0f6": "#293548",
    "#a0aabd": "#8391a6", "#e9edf5": "#243147", "#77849a": "#a8b5c9",
    "#e0e7f3": "#354762", "#e6ecfc": "#263c65", "#3155ac": "#afc6ff",
}
SETTINGS_PATH = Path(os.environ.get("APPDATA", Path.home())) / "PeppersCone" / "appearance.json"


def load_dark_mode():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8")).get("dark_mode") is True
    except (OSError, ValueError, AttributeError):
        return False


def theme_color(root, color):
    return DARK_COLORS.get(color, color) if getattr(root, "dark_mode", False) else color


def refresh_widgets(root, widget=None):
    """Refresh explicit Tk colors; ttk colors are handled by the shared styles."""
    widget = root if widget is None else widget
    originals = getattr(widget, "_studio_colors", {})
    for option in ("background", "foreground", "highlightbackground", "insertbackground"):
        if option not in widget.keys():
            continue
        value = str(widget.cget(option))
        if option not in originals:
            originals[option] = value
        original = originals[option]
        # White text on the blue brand badge stays white in either theme.
        if option == "foreground" and original == WHITE and originals.get("background") == BLUE:
            continue
        widget.configure(**{option: theme_color(root, original)})
    widget._studio_colors = originals
    if isinstance(widget, Card):
        widget.itemconfigure("surface", fill=theme_color(root, WHITE),
                             outline=theme_color(root, "#e0e5ef"))
    for child in widget.winfo_children():
        refresh_widgets(root, child)


def _rounded_buttons(root, style):
    """Nine-slice rounded borders retain ttk's native button interactions."""
    import math

    images = []

    def surface(fill, outline, surround):
        fill, outline, surround = (theme_color(root, c) for c in (fill, outline, surround))
        size, radius = 29, 12
        image = tk.PhotoImage(master=root, width=size, height=size)
        image.put(surround, to=(0, 0, size, size))
        for y in range(size):
            for x in range(size):
                dx = max(radius - x - .5, x + .5 - (size - radius), 0)
                dy = max(radius - y - .5, y + .5 - (size - radius), 0)
                distance = math.hypot(dx, dy)
                if distance <= radius:
                    edge = distance > radius - 1 or min(x, y, size-1-x, size-1-y) == 0
                    image.put(outline if edge else fill, (x, y))
        images.append(image)
        return image

    palettes = {
        "TButton": (WHITE, "#f0f4fd", "#e3ebfc", "#dce3ef", WHITE),
        "Action.TButton": (WHITE, "#f0f4fd", "#e3ebfc", "#dce3ef", BG),
        "Primary.TButton": (BLUE, "#244cce", "#1c3dae", BLUE, WHITE),
        "Action.Primary.TButton": (BLUE, "#244cce", "#1c3dae", BLUE, BG),
        "TV.TButton": (NAVY, "#283d60", "#101b30", NAVY, WHITE),
        "Action.TV.TButton": (NAVY, "#283d60", "#101b30", NAVY, BG),
    }
    if getattr(root, "dark_mode", False):
        for name in ("TV.TButton", "Action.TV.TButton"):
            values = palettes[name]
            palettes[name] = ("#30466d", *values[1:])
    root._theme_revision = getattr(root, "_theme_revision", 0) + 1
    for name, (normal, hover, pressed, border, surround) in palettes.items():
        element = f"Rounded{root._theme_revision}.{name}.border"
        base = surface(normal, border, surround)
        disabled = surface("#edf0f6", "#edf0f6", surround)
        down = surface(pressed, border, surround)
        active = surface(hover, border, surround)
        focus = surface(normal, BLUE if "Primary" not in name else "#15213a", surround)
        style.element_create(element, "image", base,
                             ("disabled", disabled), ("pressed", down),
                             ("focus", focus), ("active", active),
                             border=13, padding=0, sticky="nsew")
        style.configure(name, background=surround)
        style.map(name, background=[("disabled", surround), ("active", surround), ("!disabled", surround)])
        style.layout(name, [(element, {"sticky": "nsew", "children": [
            ("Button.padding", {"sticky": "nsew", "children": [
                ("Button.label", {"sticky": "nsew"})]})]})])
    # Quiet text actions intentionally have no surrounding button surface.
    style.layout("Quiet.TButton", [("Button.padding", {"sticky": "nsew", "children": [
        ("Button.label", {"sticky": "nsew"})]})])
    root._button_images = images  # Tcl images must stay alive for the window's lifetime.


def configure_appearance(root):
    root.configure(background=BG)
    root.option_add("*Font", "{Segoe UI} 10")
    root.option_add("*Label.Background", WHITE)
    root.option_add("*Label.Foreground", INK)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=WHITE, foreground=INK, font=("Segoe UI", 10))
    style.configure("TFrame", background=WHITE)
    style.configure("Shell.TFrame", background=BG)
    style.configure("TLabel", background=WHITE)
    style.configure("Header.TLabel", font=("Segoe UI", 19, "bold"), foreground=NAVY)
    style.configure("PageTitle.TLabel", background=BG, foreground=NAVY, font=("Segoe UI", 19, "bold"))
    style.configure("PageSubtitle.TLabel", background=BG, foreground=MUTED)
    style.configure("Body.TLabel", foreground=MUTED)
    style.configure("Quiet.TButton", borderwidth=0, foreground=MUTED, padding=(0, 6))
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("TButton", padding=(12, 8), borderwidth=1,
                    bordercolor="#dce3ef", background=WHITE, font=("Segoe UI", 10, "bold"))
    style.map("TButton", background=[("active", "#f0f4fd")],
              foreground=[("disabled", "#a0aabd")])
    for name, color in (("Primary", BLUE), ("TV", NAVY)):
        style.configure(f"{name}.TButton", background=color, foreground=WHITE, bordercolor=color)
        style.map(f"{name}.TButton", background=[("disabled", "#edf0f6"), ("active", "#244cce")],
                  foreground=[("disabled", "#a0aabd"), ("!disabled", WHITE)])
    style.configure("TLabelframe", background=WHITE, bordercolor="#e0e5ef", borderwidth=1, padding=10)
    style.configure("TLabelframe.Label", foreground=NAVY, font=("Segoe UI", 11, "bold"))
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(name, padding=7, fieldbackground="#f8f9fc", bordercolor="#e0e5ef")
        style.map(name, fieldbackground=[("readonly", "#f8f9fc")], bordercolor=[("focus", BLUE)])
    style.configure("TCheckbutton", padding=(0, 5))
    style.configure("TRadiobutton", padding=(0, 4))
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 14))
    style.configure("TNotebook.Tab", background="#e9edf5", foreground="#77849a",
                    padding=(20, 8), font=("Segoe UI", 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", WHITE), ("active", "#e0e7f3")],
              foreground=[("selected", BLUE)])
    style.configure("Preview.TLabel", background="#0f172a", foreground="#94a3b8", anchor="center")
    if getattr(root, "dark_mode", False):
        names = (".", "TFrame", "Shell.TFrame", "TLabel", "Header.TLabel",
                 "PageTitle.TLabel", "PageSubtitle.TLabel", "Body.TLabel", "Quiet.TButton",
                 "Muted.TLabel", "TButton", "Primary.TButton", "TV.TButton", "TLabelframe",
                 "TLabelframe.Label", "TEntry", "TCombobox", "TSpinbox", "TNotebook", "TNotebook.Tab")
        for name in names:
            for option, value in style.configure(name).items():
                if isinstance(value, str) and value in DARK_COLORS:
                    style.configure(name, **{option: theme_color(root, value)})
            for option, states in style.map(name).items():
                style.map(name, **{option: [(*state[:-1], theme_color(root, state[-1])) for state in states]})
        for name in ("Primary.TButton", "TV.TButton"):
            style.configure(name, foreground=WHITE)
            style.map(name, foreground=[("disabled", "#8391a6"), ("!disabled", WHITE)])
        style.configure("TCombobox", arrowcolor=INK)
    root.option_add("*TCombobox*Listbox.background", theme_color(root, WHITE))
    root.option_add("*TCombobox*Listbox.foreground", theme_color(root, INK))
    _rounded_buttons(root, style)


class Card(tk.Canvas):
    """White rounded surface with the same 16 px corners as the Mac studio."""
    def __init__(self, parent, padding=14):
        super().__init__(parent, background=BG, highlightthickness=0, width=1, height=1)
        self.content = ttk.Frame(self, padding=padding)
        window = self.create_window(8, 8, anchor="nw", window=self.content)

        def draw(event):
            w, h, r = event.width - 1, event.height - 1, 16
            self.delete("surface")
            self.create_polygon(r, 0, w-r, 0, w, 0, w, r, w, h-r,
                                w, h, w-r, h, r, h, 0, h, 0, h-r,
                                0, r, 0, 0, smooth=True, fill=theme_color(self.winfo_toplevel(), WHITE),
                                outline=theme_color(self.winfo_toplevel(), "#e0e5ef"), tags="surface")
            self.tag_lower("surface")
            self.itemconfigure(window, width=max(1, w-16), height=max(1, h-16))
        self.bind("<Configure>", draw)


class ScrollPanel(ttk.Frame):
    """A bounded control column with local mouse-wheel scrolling."""
    def __init__(self, parent, width=440):
        super().__init__(parent, style="Shell.TFrame")
        self.canvas = tk.Canvas(self, width=width, background=WHITE,
                                highlightthickness=1, highlightbackground="#e0e5ef")
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content = ttk.Frame(self.canvas, padding=16)
        window = self.canvas.create_window(0, 0, anchor="nw", window=self.content)
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width))
        self.bind("<Map>", self._bind_wheel)

    def _bind_wheel(self, event=None):
        def bind(widget):
            widget.bind("<MouseWheel>", self._wheel, add="+")
            for child in widget.winfo_children():
                bind(child)
        bind(self)

    def _wheel(self, event):
        if self.content.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(-int(event.delta / 120), "units")
        return "break"


class StudioApp(tk.Tk):
    def __init__(self, circle=False):
        super().__init__()
        self.title("Pepper's Cone Studio" + (" — Circle" if circle else ""))
        self.geometry("1280x850")
        self.minsize(1000, 720)
        self.dark_mode = load_dark_mode()
        configure_appearance(self)
        shell = ttk.Frame(self, style="Shell.TFrame", padding=(24, 20))
        shell.pack(fill="both", expand=True)
        header = tk.Frame(shell, background=BG)
        header.pack(fill="x", pady=(0, 20))
        tk.Label(header, text="PC", bg=BLUE, fg=WHITE, font=("Segoe UI", 12, "bold"),
                 padx=9, pady=8).pack(side="left", padx=(0, 12))
        brand = tk.Frame(header, background=BG)
        brand.pack(side="left")
        tk.Label(brand, text="Pepper's Cone", bg=BG, fg="#15213a", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(brand, text="DISPLAY STUDIO", bg=BG, fg="#7b8799", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(header, text="CIRCLE EDITION" if circle else "WINDOWS EDITION", bg="#e6ecfc",
                 fg="#3155ac", padx=12, pady=6, font=("Segoe UI", 9, "bold")).pack(side="right")
        self.theme_button = ttk.Button(header, command=self.toggle_dark_mode)
        self.theme_button.pack(side="right", padx=12)
        self.theme_button.configure(text="Light mode" if self.dark_mode else "Dark mode")
        navigation = ttk.Frame(shell, style="Shell.TFrame")
        navigation.pack(fill="x", pady=(0, 16))
        self.navigation_buttons = {}
        self.page_stack = ttk.Frame(shell, style="Shell.TFrame")
        self.page_stack.pack(fill="both", expand=True)
        self.page_stack.rowconfigure(0, weight=1)
        self.page_stack.columnconfigure(0, weight=1)
        self._selected_page = "LiveView"
        self.pages = {}
        self.hosts = {}
        self._loading = set()
        self._page_errors = {}
        suffix = "_circle" if circle else ""
        self.specs = [("LiveView", "live_view" + suffix, "Live"),
                      ("RecordView", "record_view" + suffix, "Record"),
                      ("UploadView", "upload_view" + suffix, "Upload")]
        if not circle:
            self.specs.append(("CalibrationView", "calibration_view", "Calibrate"))
        for name, module, title in self.specs:
            button = ttk.Button(navigation, text=title, style="Action.TButton",
                                command=lambda selected=name: self.show_page(selected))
            button.pack(side="left", padx=(0, 10))
            self.navigation_buttons[name] = button
            host = ttk.Frame(self.page_stack, style="Shell.TFrame", padding=(0, 8))
            host.grid(row=0, column=0, sticky="nsew")
            self.hosts[name] = host
        refresh_widgets(self)
        self.after_idle(lambda: self.show_page("LiveView"))

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        configure_appearance(self)
        refresh_widgets(self)
        self.theme_button.configure(text="Light mode" if self.dark_mode else "Dark mode")
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps({"dark_mode": self.dark_mode}), encoding="utf-8")
        except OSError:
            # A read-only profile must not prevent changing the current session.
            pass

    def _load_selected(self, event=None):
        name, module, _ = next(spec for spec in self.specs if spec[0] == self._selected_page)
        if name in self.pages or name in self._loading or name in self._page_errors:
            return
        host = self.hosts[name]
        self._loading.add(name)
        try:
            page_class = getattr(importlib.import_module(module), name)
            page = page_class(host, self)
            page.pack(fill="both", expand=True)
            self.pages[name] = page
        except Exception as error:
            traceback.print_exc()
            # A constructor can fail after creating widgets; discard that partial page.
            for child in host.winfo_children():
                child.destroy()
            panel = ttk.Frame(host, padding=24)
            panel.pack(fill="both", expand=True)
            self._page_errors[name] = panel
            ttk.Label(panel, text="This page could not start", style="Header.TLabel").pack(anchor="w")
            ttk.Label(panel, text=f"{type(error).__name__}: {error}",
                      wraplength=760, style="Body.TLabel").pack(anchor="w", pady=16)
            ttk.Label(panel, text="Check the Python environment used to launch the studio, then retry. "
                      "The other tabs are still available.", wraplength=760).pack(anchor="w")
            ttk.Button(panel, text="Retry", style="Primary.TButton",
                       command=lambda: self._retry_page(name)).pack(anchor="w", pady=16)
        finally:
            self._loading.discard(name)
            refresh_widgets(self)

    def _retry_page(self, name):
        panel = self._page_errors.pop(name, None)
        if panel is not None:
            panel.destroy()
        self.show_page(name)

    def show_page(self, name):
        # Existing Back callbacks return to the main Live tab.
        name = "LiveView" if name == "HomePage" else name
        # Pages that hold hardware (e.g. the calibration camera) release it when left.
        previous = self.pages.get(self._selected_page)
        if self._selected_page != name and hasattr(previous, "on_hide"):
            previous.on_hide()
        self._selected_page = name
        self.hosts[name].tkraise()
        for page_name, button in self.navigation_buttons.items():
            button.configure(style="Action.Primary.TButton" if page_name == name else "Action.TButton")
        self._load_selected()
