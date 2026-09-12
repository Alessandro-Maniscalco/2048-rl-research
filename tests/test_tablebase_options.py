from pathlib import Path
import pytest

from research.tablebase_pilot import native_command


def test_explicit_probability_override_follows_depth_and_preserves_default():
    original = ['/tmp/vendor/2048', '-d', '5', '-i', '1', '-v', '123']
    assert native_command(Path('/tmp/vendor'), 5, 123) == original
    explicit = native_command(Path('/tmp/vendor'), 5, 123, 1/4096)
    assert explicit[:-3] == original[:-1]
    assert explicit[-3:] == ['-p', '0.000244140625', '123']
    for invalid in (0, -1, 2, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='Probability cutoff'):
            native_command('/tmp/vendor', 5, 123, invalid)


def test_optional_lookup_threshold_is_explicit_and_preserves_original_default():
    assert '-t' not in native_command('/tmp/vendor', 5, 123)
    assert native_command('/tmp/vendor', 5, 123, lookup_probability=.01)[-3:] == ['-t', '0.01', '123']
    assert native_command('/tmp/vendor', 5, 123, lookup_probability=0.)[-3:] == ['-t', '0.0', '123']
    for invalid in (-1., 1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='Lookup threshold'):
            native_command('/tmp/vendor', 5, 123, lookup_probability=invalid)
