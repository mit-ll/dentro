"""
Nox creates a new virtual environment for each individual test.  Thus, it is important for to install all the packages needed for testing.  When using Nox, it will by default grab the current python version available in your environment and run testing with it.

Useful commands:

```console
nox --list                 # Lists out all the available sessions
nox -r -s pytest           # Run pytests
nox -r -s pytest_cov       # Run pytests with coverage report
nox -r -s coverage         # Run coverage (stand-alone)
nox -r -s scalene          # Profile the code
nox -r -s pdoc             # Generate documentation
nox -r -s show_pdoc        # View HTML of pdoc

nox                        # Run all sessions
```

"""
from dataclasses import dataclass

import nox

from src.ci.utils import view_html


@dataclass
class config:
    pytest_cov_path: str = "save/pytest-cov"
    coverage_path: str = "save/coverage"
    pdoc_path: str = "save/pdocs"


@nox.session
def pytest(session: nox.Session):
    """Run PyTests."""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("pytest", "-v")


@nox.session
def pytest_cov(session: nox.Session):
    """Run PyTests with coverage."""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("pytest", "--cov=./", f"--cov-report=html:{config.pytest_cov_path}")


@nox.session
def coverage(session: nox.Session):
    """Runs coverage pytests"""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("coverage", "run", "-m", "pytest")
    session.run("coverage", "html", "-d", config.coverage_path)
    session.run("coverage", "report", "-m")


@nox.session
def scalene(session: nox.Session):
    """Profiles your selected code using scalene."""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("scalene", "-m", "pytest")


@nox.session
def pdoc(session: nox.Session):
    """Generate pdocs."""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("mkdir", "-p", f"{config.pdoc_path}/docs")
    session.run("cp", "-rf", "docs/pics", f"{config.pdoc_path}/docs/")
    session.run(
        "pdoc",
        "-d",
        "google",
        "--logo",
        "https://github.com/mit-ll/dentro/blob/main/docs/pics/program_logo.png?raw=true",
        "--logo-link",
        "https://github.com/mit-ll/dentro",
        "--math",
        "--mermaid",
        "--footer-text",
        "Author: W. Li",
        "--output-directory",
        config.pdoc_path,
        "src",
    )


@nox.session
def show_pytest_cov(session: nox.Session):
    """Show pytest coverage in HTML."""

    pytest_cov(session)
    view_html(config.pdoc_path)


@nox.session
def show_pdoc(session: nox.Session):
    """Show pdoc in HTML."""

    pdoc(session)
    view_html(config.pdoc_path)
