import pytest

from speaker_transcribe.chunking import (
    StalledError,
    Unit,
    format_ladder,
    merge_regions,
    parse_ladder,
    run_with_fallback,
    split_region,
)


def test_parse_ladder_reads_minutes_as_seconds():
    assert parse_ladder("5, 2, 1") == (300.0, 120.0, 60.0)
    assert parse_ladder("5 2;1") == (300.0, 120.0, 60.0)
    assert parse_ladder("0.5") == (30.0,)
    assert parse_ladder("  ") == ()


@pytest.mark.parametrize("text", ["5, x", "-2", "0"])
def test_parse_ladder_rejects_non_positive_numbers(text):
    with pytest.raises(ValueError):
        parse_ladder(text)


def test_format_ladder_round_trips():
    assert format_ladder((300.0, 120.0, 30.0)) == "5, 2, 0.5"
    assert parse_ladder(format_ladder((300.0, 90.0))) == (300.0, 90.0)


def test_split_region_covers_the_span_without_gaps():
    spans = split_region(0.0, 1447.0, 300.0)
    assert spans[0] == (0.0, 300.0)
    assert spans[-1][1] == 1447.0
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert end == start


def test_split_region_folds_a_tiny_tail_into_the_last_chunk():
    assert split_region(0.0, 300.4, 300.0) == [(0.0, 300.4)]
    assert split_region(0.0, 60.0, 300.0) == [(0.0, 60.0)]
    assert split_region(10.0, 10.0, 300.0) == []


def test_merge_regions_joins_touching_spans():
    assert merge_regions([(120.0, 180.0), (0.0, 60.0), (60.0, 120.0)]) == [(0.0, 180.0)]
    assert merge_regions([(0.0, 60.0), (120.0, 180.0)]) == [(0.0, 60.0), (120.0, 180.0)]


def _runner(stall_above: float, calls: list[Unit]):
    """A fake engine that stalls on any unit longer than `stall_above` seconds."""

    def run_unit(unit: Unit) -> list[str]:
        calls.append(unit)
        if unit.duration > stall_above:
            raise StalledError("no progress")
        return [f"{unit.start:.0f}-{unit.end:.0f}"]

    return run_unit


def test_whole_file_is_transcribed_in_one_call_when_nothing_stalls():
    calls: list[Unit] = []
    out = run_with_fallback(600.0, (300.0, 120.0), _runner(10_000, calls))

    assert out == ["0-600"]
    assert [unit.chunk_size for unit in calls] == [None]


def test_a_stalled_run_is_retried_in_chunks():
    calls: list[Unit] = []
    out = run_with_fallback(900.0, (300.0, 120.0), _runner(300.0, calls))

    assert out == ["0-300", "300-600", "600-900"]
    # whole file first, then one 5-minute chunk per third of the audio
    assert [unit.chunk_size for unit in calls] == [None, 300.0, 300.0, 300.0]


def test_only_the_stretch_that_keeps_stalling_is_cut_smaller():
    calls: list[Unit] = []

    def run_unit(unit: Unit) -> list[str]:
        calls.append(unit)
        if unit.duration > 300.0:  # the whole file
            raise StalledError("no progress")
        if unit.start == 300.0 and unit.duration > 120.0:  # the middle chunk
            raise StalledError("no progress")
        return [f"{unit.start:.0f}-{unit.end:.0f}"]

    out = run_with_fallback(900.0, (300.0, 120.0), run_unit)

    assert out == ["0-300", "600-900", "300-420", "420-540", "540-600"]
    retried = [(unit.start, unit.end) for unit in calls if unit.chunk_size == 120.0]
    assert retried == [(300.0, 420.0), (420.0, 540.0), (540.0, 600.0)]


def test_chunking_can_start_without_a_whole_file_attempt():
    calls: list[Unit] = []
    run_with_fallback(600.0, (300.0,), _runner(300.0, calls), whole_file_first=False)

    assert [unit.chunk_size for unit in calls] == [300.0, 300.0]


def test_audio_that_stalls_at_every_size_is_skipped_with_a_warning():
    logs: list[str] = []
    calls: list[Unit] = []

    def run_unit(unit: Unit) -> list[str]:
        calls.append(unit)
        if unit.start < 300.0:
            raise StalledError("no progress")
        return [f"{unit.start:.0f}-{unit.end:.0f}"]

    out = run_with_fallback(600.0, (300.0, 60.0), run_unit, log=logs.append)

    assert out == ["300-600"]
    assert any("Skipped 0:00–5:00" in line for line in logs)


def test_skip_failed_off_turns_unrecoverable_audio_into_an_error():
    def run_unit(unit: Unit) -> list[str]:
        if unit.start < 300.0:
            raise StalledError("no progress")
        return ["ok"]

    with pytest.raises(StalledError, match="0:00–5:00"):
        run_with_fallback(600.0, (300.0, 60.0), run_unit, skip_failed=False)


def test_a_failure_that_chunking_cannot_fix_is_raised_as_is():
    calls: list[Unit] = []

    def run_unit(unit: Unit) -> list[str]:
        calls.append(unit)
        raise RuntimeError("model 'large-v3' not found")

    with pytest.raises(RuntimeError, match="not found"):
        run_with_fallback(900.0, (300.0, 120.0, 60.0), run_unit)

    # whole file, then the 5-minute level — the ladder stops there instead of
    # repeating the same error for every 2- and 1-minute chunk
    assert [unit.chunk_size for unit in calls] == [None, 300.0, 300.0, 300.0]


def test_rungs_shorter_than_the_minimum_are_ignored():
    calls: list[Unit] = []
    run_with_fallback(600.0, (300.0, 1.0), _runner(300.0, calls))

    assert {unit.chunk_size for unit in calls} == {None, 300.0}
