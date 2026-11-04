"""Launch the Pepper's Cone display studio."""
from studio_theme import StudioApp


class App(StudioApp):
    def __init__(self):
        super().__init__(circle=True)


if __name__ == "__main__":
    App().mainloop()
