"""
Nox creates a new virtual environment for each individual test.  Thus, it is important for to install all the packages needed for testing.  When using Nox, it will by default grab the current python version available in your environment and run testing with it.

Useful commands:

```console
nox --list              # Lists out all the available sessions
nox -s pytest           # Run pytests
nox -s coverage         # Run coverage
nox -s profile          # Profile the code
nox -s autodoc          # Generate documentation

nox                     # Run all sessions
```

"""
import nox

from src.ci.utils import autodoc as ci_autodoc
from src.ci.utils import coverage as ci_coverage
from src.ci.utils import profile as ci_profile


@nox.session(python=["3.8", "3.9", "3.10"])
def pytest(session):
    """Run PyTests."""

    session.run("poetry", "install", "--with=dev", external=True)
    session.run("pytest", "-v")


@nox.session
def coverage(session):
    """Runs coverage pytests"""

    session.run("poetry", "install", "--with=dev", external=True)
    ci_coverage()


@nox.session
def profile(session):
    """Profiles your selected code using scalene."""

    session.run("poetry", "install", "--with=dev", external=True)
    ci_profile()


@nox.session
def autodoc(session):
    """Generate pdocs."""

    session.run("poetry", "install", "--with=dev", external=True)
    ci_autodoc()
