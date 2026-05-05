import pytest
from crowdsegmenter.utils.formatter import format_time

def test_format_time():
    assert format_time(0) == "00:00"
    assert format_time(59) == "00:59"
    assert format_time(60) == "01:00"
    assert format_time(65) == "01:05"
    assert format_time(3600) == "60:00"
    assert format_time(3665) == "61:05"

def test_format_time_with_floats():
    assert format_time(65.4) == "01:05"
    assert format_time(65.9) == "01:05"
