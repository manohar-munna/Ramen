"""Finding Chrome, and taking a screenshot with it.

Every measuring tool here works the same way: render the reconstruction to a file,
photograph it in headless Chrome, compare the photograph with the original. Four of
them had grown their own copy of that, and two of those had stopped looking for Chrome
at all -- a single absolute path into Program Files, which fails on a machine with the
32-bit install, with Chrome somewhere else, or on anything that is not Windows. The
failure is a FileNotFoundError from subprocess with no hint as to what is missing.
"""
import os
import subprocess

# Same list the enhancer uses. The bare names are for PATH lookup on Linux.
CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
)


def find_chrome(explicit=None):
    """The Chrome binary, or a message saying what was looked for and how to say it."""
    import shutil
    for c in ([explicit] if explicit else []) + list(CANDIDATES):
        if os.path.isfile(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    raise RuntimeError(
        "Headless Chrome is needed to render the reconstruction and none was found. "
        "Looked for: %s. Set CHROME to the binary to use."
        % ", ".join(CANDIDATES))


def screenshot(chrome, html_path, png_path, width, height,
               scale=1.0, extra=(), timeout=180):
    """Photograph a local HTML file at an exact size. Paths may be relative."""
    html_path = os.path.abspath(html_path)
    png_path = os.path.abspath(png_path)
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
         "--force-device-scale-factor=%s" % scale,
         # Rounded, not truncated: an A4 page is 841.89pt tall and the difference
         # between 841 and 842 is a row of pixels the comparison then counts as wrong.
         "--window-size=%d,%d" % (round(width), round(height)),
         "--screenshot=" + png_path]
        + list(extra)
        + ["file:///" + html_path.replace(os.sep, "/")],
        check=True, capture_output=True, timeout=timeout)
    return png_path
