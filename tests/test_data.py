from pathlib import Path

import numpy as np

from digital_twin.data import (
    infer_effective_time,
    parse_experiment_id,
    read_interrogator_file,
    safe_output_stem,
)


def test_parse_experiment_id_filename_variants():
    cases = {
        "19 cm-16 layers-7-s-interrogator.txt": "19cm-16layers-7-s",
        "19cm-16-layers-1-s-interrogator.txt": "19cm-16layers-1-s",
        "23cm-16-layers-1-interrogator.txt": "23cm-16layers-1",
        "15cm-12layers-8-interrogator.txt": "15cm-12layers-8",
    }
    for name, expected in cases.items():
        assert parse_experiment_id(name).normalized_stem == expected


def test_safe_output_stem_keeps_duplicate_variants_distinct():
    assert safe_output_stem("15cm-16-layers-2-s-interrogator.txt") != safe_output_stem(
        "15cm-16layers-2-s-interrogator.txt"
    )


def test_effective_time_spreads_repeated_seconds():
    effective = infer_effective_time([10, 10, 10, 11, 11])
    np.testing.assert_allclose(effective, [10.0, 10.333333, 10.666667, 11.0, 11.5])


def test_read_interrogator_file_infers_channels_from_rows(tmp_path: Path):
    path = tmp_path / "15cm-16-layers-2-s-interrogator.txt"
    path.write_text(
        "\n".join(
            [
                "Timestamp\tTime [s]\tWL 1[nm]\tWL 2[nm]\tWL 3[nm]\tWL 4[nm]",
                "2025-01-01T00:00:00Z\t1.00000\t1550.0\t1530.0",
                "2025-01-01T00:00:00Z\t1.00000\t1550.1\t1530.1\t1520.1",
            ]
        ),
        encoding="utf-8",
    )
    frame = read_interrogator_file(path)
    assert list(frame.columns) == [
        "timestamp",
        "time_s",
        "effective_time_s",
        "sample_index",
        "wl_1_nm",
        "wl_2_nm",
        "wl_3_nm",
    ]
    assert frame["wl_3_nm"].isna().iloc[0]
    assert frame.attrs["normalized_stem"] == "15cm-16layers-2-s"
