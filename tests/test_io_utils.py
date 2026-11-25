from __future__ import annotations

import json

import pytest

from windpower import io_utils as U


def test_atomic_write_roundtrip_and_no_temp_left(tmp_path):
    p = U.atomic_write_json(tmp_path / "sub" / "a.json", {"b": 1, "ä": "x"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"b": 1, "ä": "x"}
    assert [f.name for f in p.parent.iterdir()] == ["a.json"]


def test_atomic_write_failure_keeps_old_file(tmp_path):
    p = U.atomic_write_text(tmp_path / "f.txt", "old")

    def boom(tmp):
        tmp.write_text("partial")
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        U.atomic_write_with(p, boom)
    assert p.read_text() == "old"
    assert [f.name for f in tmp_path.iterdir()] == ["f.txt"]


def test_sha256_known_vector(tmp_path):
    p = U.atomic_write_bytes(tmp_path / "x", b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert U.sha256_file(p) == expected == U.sha256_bytes(b"abc")


def test_utc_helpers():
    assert str(U.to_utc_naive_index(["2025-07-01T03:00:00+03:00"])[0]) == "2025-07-01 00:00:00"
    assert str(U.to_utc_naive_index(["2025-07-01T00:00:00"])[0]) == "2025-07-01 00:00:00"
    assert len(U.utc_hour_range("2025-07-01T00:00:00", "2025-09-30T23:00:00")) == 2208
    assert U.epoch_ms_to_iso(0) == "1970-01-01T00:00:00Z"
    assert U.epoch_ms_to_iso(None) is None
    assert U.iso_utc("2025-07-01T00:00:00") == "2025-07-01T00:00:00Z"


def test_table_schema_units():
    import pandas as pd

    s = U.table_schema(pd.DataFrame({"capacity_mw": [1.0], "mystery": [1]}))
    assert s["capacity_mw"] == {"dtype": "float64", "unit": "MW"}
    assert s["mystery"]["unit"] is None
