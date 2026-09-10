from mimicneoai.functions.immunogenicity_runner import resolve_immunogenicity_num_processes


def test_both_pipeline_thread_spellings_are_supported():
    assert resolve_immunogenicity_num_processes({"args": {"thread": 20}}) == 20
    assert resolve_immunogenicity_num_processes({"args": {"threads": 30}}) == 30


def test_missing_pipeline_thread_uses_conservative_standalone_default():
    assert resolve_immunogenicity_num_processes({"args": {}}) == 1


def test_invalid_pipeline_thread_is_rejected():
    for value in (0, -1, "bad"):
        try:
            resolve_immunogenicity_num_processes({"args": {"threads": value}})
        except ValueError as exc:
            assert "threads" in str(exc)
        else:
            raise AssertionError(f"Expected invalid worker value to fail: {value!r}")
