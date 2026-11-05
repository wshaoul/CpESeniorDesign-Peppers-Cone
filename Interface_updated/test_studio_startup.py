"""Hardware-free regressions for the tab startup and failure paths.

Run with: python -m unittest discover -s Interface_updated -p test_studio_startup.py
"""
import contextlib
import io
import types
import unittest
from unittest.mock import patch
from tkinter import ttk

from studio_theme import StudioApp


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.app = StudioApp()
        self.app.withdraw()
        self.callback_errors = []
        self.app.report_callback_exception = lambda *args: self.callback_errors.append(args)

    def tearDown(self):
        self.app.destroy()

    def test_initial_page_loads_without_manual_navigation(self):
        class Page(ttk.Frame):
            def __init__(self, parent, controller):
                super().__init__(parent)
                ttk.Label(self, text="Live controls").pack()

        with patch("studio_theme.importlib.import_module", return_value=types.SimpleNamespace(LiveView=Page)):
            self.app.update()
        self.assertIn("LiveView", self.app.pages)
        self.assertEqual(self.app.pages["LiveView"].winfo_manager(), "pack")
        self.assertFalse(self.callback_errors)

    def test_missing_dependency_shows_error_and_retry_recovers(self):
        with patch("studio_theme.importlib.import_module", side_effect=ModuleNotFoundError("No module named 'cv2'")):
            with contextlib.redirect_stderr(io.StringIO()):
                self.app.update()
        panel = self.app._page_errors["LiveView"]
        labels = [child.cget("text") for child in panel.winfo_children()]
        self.assertTrue(any("cv2" in label for label in labels))
        self.assertEqual(panel.winfo_manager(), "pack")
        self.assertFalse(self.callback_errors)

        class Page(ttk.Frame):
            def __init__(self, parent, controller):
                super().__init__(parent)

        with patch("studio_theme.importlib.import_module", return_value=types.SimpleNamespace(LiveView=Page)):
            self.app._retry_page("LiveView")
            self.app.update()
        self.assertIn("LiveView", self.app.pages)
        self.assertNotIn("LiveView", self.app._page_errors)

    def test_constructor_failure_is_cleaned_up_and_other_tabs_work(self):
        class BrokenPage(ttk.Frame):
            def __init__(self, parent, controller):
                super().__init__(parent)
                ttk.Label(self, text="Partial page").pack()
                raise RuntimeError("Initialization failed")

        class WorkingPage(ttk.Frame):
            def __init__(self, parent, controller):
                super().__init__(parent)

        with patch("studio_theme.importlib.import_module", return_value=types.SimpleNamespace(
                LiveView=BrokenPage, RecordView=WorkingPage)):
            with contextlib.redirect_stderr(io.StringIO()):
                self.app.update()
            self.assertEqual(len(self.app.hosts["LiveView"].winfo_children()), 1)
            self.app.show_page("RecordView")
            self.app.update()
        self.assertIn("RecordView", self.app.pages)
        self.assertFalse(self.callback_errors)

    def test_calibrate_tab_is_windows_only(self):
        class Page(ttk.Frame):
            def __init__(self, parent, controller):
                super().__init__(parent)

        self.assertIn("CalibrationView", self.app.navigation_buttons)
        with patch("studio_theme.importlib.import_module", return_value=types.SimpleNamespace(LiveView=Page)):
            self.app.update()  # run the pending startup callback before destroying the root
            self.app.destroy()  # one Tk root at a time; tearDown destroys the replacement
            self.app = StudioApp(circle=True)
            self.app.withdraw()
            self.app.update()
        self.assertNotIn("CalibrationView", self.app.navigation_buttons)

    def test_leaving_a_page_calls_on_hide(self):
        hidden = []

        class Page(ttk.Frame):
            def __init__(self, parent, controller):
                super().__init__(parent)

            def on_hide(self):
                hidden.append(True)

        with patch("studio_theme.importlib.import_module", return_value=types.SimpleNamespace(
                LiveView=Page, CalibrationView=Page)):
            self.app.update()
            self.app.show_page("CalibrationView")
            self.app.show_page("CalibrationView")
            self.assertEqual(len(hidden), 1)  # left Live once; re-selecting Calibrate is not a hide
            self.app.show_page("HomePage")
        self.assertEqual(len(hidden), 2)
        self.assertFalse(self.callback_errors)


if __name__ == "__main__":
    unittest.main()
