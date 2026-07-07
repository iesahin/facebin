"""Tests for shared server utilities and the error hierarchy."""

import numpy as np

from facebin.errors import (CameraError, ConfigError, DatabaseError,
                            FacebinError, RedisConnectionError)
from facebin.server.utils import init_logging, shuffle_parallel, static_vars


def test_init_logging_returns_singleton():
    assert init_logging() is init_logging()


def test_static_vars():
    @static_vars(counter=0)
    def bump():
        bump.counter += 1
        return bump.counter

    assert bump() == 1
    assert bump() == 2


def test_shuffle_parallel_keeps_rows_aligned():
    a = np.arange(100)
    b = np.arange(100) * 10
    shuffle_parallel(a, b)
    np.testing.assert_array_equal(a * 10, b)
    # And it actually shuffles (astronomically unlikely to be identity).
    assert not np.array_equal(a, np.arange(100))


def test_errors_are_facebin_errors():
    for cls in (ConfigError, DatabaseError, RedisConnectionError,
                CameraError):
        err = cls("boom")
        assert isinstance(err, FacebinError)
        assert str(err) == "boom"


def test_error_hint_is_shown():
    err = ConfigError("bad value", hint="fix it in facebin.toml")
    text = str(err)
    assert "bad value" in text
    assert "Hint: fix it in facebin.toml" in text
