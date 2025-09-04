from __future__ import annotations

import hh4b_transformer as m


def test_version():
    assert isinstance(m.__version__, str)
