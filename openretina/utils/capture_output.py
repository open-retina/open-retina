import contextlib
import io
import warnings

from IPython.core.interactiveshell import InteractiveShell


class CaptureOutputAndWarnings:
    def __enter__(self):
        # Redirect stdout to capture print statements
        self._stdout = io.StringIO()
        self._original_stdout = contextlib.redirect_stdout(self._stdout)
        self._original_stdout.__enter__()

        # Capture and suppress stderr (tqdm progress bars and warnings)
        self._stderr = io.StringIO()
        self._original_stderr = contextlib.redirect_stderr(self._stderr)
        self._original_stderr.__enter__()

        # Capture warnings
        self._original_warnings = warnings.catch_warnings()
        self._original_warnings.__enter__()
        warnings.simplefilter("ignore")

        # Suppress IPython display output (tqdm.notebook and tqdm.auto)
        self._display_pub = InteractiveShell.instance().display_pub
        self._original_publish_method = self._display_pub.publish
        self._display_pub.publish = self._noop

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore stdout
        self._original_stdout.__exit__(exc_type, exc_value, traceback)
        self._stdout.close()

        # Restore stderr
        self._original_stderr.__exit__(exc_type, exc_value, traceback)
        self._stderr.close()

        # Stop ignoring warnings
        self._original_warnings.__exit__(exc_type, exc_value, traceback)

        # Restore IPython display method
        self._display_pub.publish = self._original_publish_method

    def _noop(self, *args, **kwargs):
        """A no-op display function to suppress IPython outputs."""
        pass
