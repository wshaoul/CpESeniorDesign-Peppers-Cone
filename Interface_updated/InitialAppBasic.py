import tkinter as tk
from tkinter import ttk

APP_TITLE = "Live & Upload — Stub"
APP_WIDTH, APP_HEIGHT = 520, 360

class StudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.minsize(420, 300)
        self.configure(bg="#f7f7fb")

        # Container for simple page switching
        self.container = ttk.Frame(self, padding=20)
        self.container.pack(fill="both", expand=True)

        self.pages = {}
        for Page in (HomePage, LiveStub, UploadStub):
            page = Page(parent=self.container, controller=self)
            self.pages[Page.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page("HomePage")

        # Simple styling
        style = ttk.Style(self)
        style.configure("TButton", padding=10)
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Body.TLabel", font=("Segoe UI", 10))

    def show_page(self, name):
        page = self.pages[name]
        page.tkraise()

class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        header = ttk.Label(self, text="Live & Upload Studio", style="Header.TLabel")
        subtitle = ttk.Label(self, text="Choose what you want to do:", style="Body.TLabel")

        buttons = ttk.Frame(self)
        live_btn = ttk.Button(buttons, text="🎥  Live Stream", command=lambda: controller.show_page("LiveStub"))
        upload_btn = ttk.Button(buttons, text="📤  Upload Video", command=lambda: controller.show_page("UploadStub"))

        header.pack(pady=(10, 4))
        subtitle.pack(pady=(0, 16))
        buttons.pack()

        live_btn.grid(row=0, column=0, padx=8, pady=8, ipadx=20, ipady=10)
        upload_btn.grid(row=0, column=1, padx=8, pady=8, ipadx=20, ipady=10)

class LiveStub(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        title = ttk.Label(self, text="Live Stream (stub)", style="Header.TLabel")
        desc = ttk.Label(self, text="This is a placeholder screen.\nHook up real live-stream logic later.", style="Body.TLabel")
        back = ttk.Button(self, text="← Back", command=lambda: controller.show_page("HomePage"))

        title.pack(pady=(10, 8))
        desc.pack(pady=(0, 20))
        back.pack()

class UploadStub(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        title = ttk.Label(self, text="Upload Video (stub)", style="Header.TLabel")
        desc = ttk.Label(self, text="This is a placeholder screen.\nHook up real upload logic later.", style="Body.TLabel")
        back = ttk.Button(self, text="← Back", command=lambda: controller.show_page("HomePage"))

        title.pack(pady=(10, 8))
        desc.pack(pady=(0, 20))
        back.pack()

if __name__ == "__main__":
    app = StudioApp()
    app.mainloop()
