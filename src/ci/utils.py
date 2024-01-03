"""
Common functions that are executed when running Continuous Integration (CI) include:

* `pytests`: tests to verify code
* `coverage`: coverage of code given pytests
* `profile`: profiling of certain functions
* `autodoc`: automatic documentation generation
"""
import os
import subprocess
import webbrowser


def coverage(browser: str | None = None, local_path: str = "save/coverage"):
    """Run coverage of code based on pytests.

    Args:
        browser (str | None, optional): Possible arguments include `Chrome`, `Firefox`, `Safari`.  If no browser is specified it will open using your system's default browser. Defaults to None.
        local_path (str, optional): The local path to save the files. Defaults to "save/coverage".
    """
    # Run coverage using pytest, then record results to docs.
    subprocess.run(["coverage", "run", "-m", "pytest"])
    subprocess.run(["coverage", "html", "-d", local_path])
    subprocess.run(["coverage", "report", "-m"])

    # Open in a browser and view results
    cwd = os.getcwd()
    url = f"file://{cwd}/{local_path}/index.html"
    webbrowser.get(browser).open(url)


def profile():
    """Profiles your selected code using scalene.  Works with `multiprocessing` package but not `Ray`."""

    subprocess.run(["scalene", "-m", "pytest"])


def autodoc(browser: str | None = None, local_path: str = "save/pdocs"):
    """Generate automatic documentation.

    Args:
        browser (str | None, optional): Possible arguments include `Chrome`, `Firefox`, `Safari`.  If no browser is specified it will open using your system's default browser. Defaults to None.
        local_path (str, optional): The local path to save the files. Defaults to "save/pdocs".
    """

    subprocess.run(["mkdir", "-p", f"{local_path}/docs"])
    subprocess.run(["cp", "-rf", "docs/pics", f"{local_path}/docs/"])
    subprocess.run(
        [
            "pdoc",
            "-d",
            "google",
            "--logo",
            "https://github.com/destin-v/dentro/blob/main/docs/pics/program_logo.png?raw=true",
            "--logo-link",
            "https://github.com/destin-v/dentro",
            "--footer-text",
            "Author: W. Li",
            "--output-directory",
            local_path,
            "src",
        ]
    )

    # Open in a browser and view results
    cwd = os.getcwd()
    url = f"file://{cwd}/{local_path}/index.html"
    webbrowser.get(browser).open(url)
