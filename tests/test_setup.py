import boa

import lcom


def test_package_imports():
    assert lcom.__doc__


def test_boa_is_the_pinned_version():
    assert boa.__version__ == "1.10.0"
