from .base_solution import BaseSolution


class Face(BaseSolution):
    pass


class FacesDetector(object):
    def __init__(self):
        pass

    def process(self):
        pass

    def __enter__(self):
        """A "with" statement support."""
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        """A "with" statement support."""
        if exc_type is not None:
            print(exc_type, exc_val, traceback)  # traceback.(tb_frame, tb_lasti, tb_lineno, tb_next)
            return False
        return self
