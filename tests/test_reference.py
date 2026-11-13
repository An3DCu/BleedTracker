from bleedtracker.config import FEATURES
from bleedtracker.reference import OLD_NAMES, load_reference, rename
from bleedtracker.rule import POSITION_BANDS


def test_nested_keys_get_new_names():
    old = {"anchor": {"bands": {"1-3": 0.1, "21+": 0.0}, "rows": [{"pos_mar": 2.0}]}}
    assert rename(old) == {
        "anchor": {"bands": {"up to 3": 0.1, "over 20": 0.0}, "rows": [{"position": 2.0}]}
    }


def test_values_and_unknown_keys_stay_as_they_are():
    assert rename({"note": "1-3", "slice_pages": 85453}) == {"note": "1-3", "slice_pages": 85453}


def test_every_new_name_is_used_by_the_package():
    assert set(OLD_NAMES.values()) <= set(FEATURES) | set(POSITION_BANDS)


def test_reference_file_reads_with_new_band_names():
    bands = load_reference()["anchor"]["band_median_ctr_percent"]
    assert list(bands) == POSITION_BANDS
