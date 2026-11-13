from importlib.metadata import version

import bleedtracker


def test_version_matches_the_installed_package():
    assert bleedtracker.__version__ == version("bleedtracker")
