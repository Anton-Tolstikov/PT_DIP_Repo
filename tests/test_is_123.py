import pytest
from is_123 import is_123


def test_is_123_true():
    assert is_123("123") is True


def test_is_123_false_wrong_number():
    assert is_123("124") is False


def test_is_123_false_non_numeric():
    assert is_123("abc") is False
