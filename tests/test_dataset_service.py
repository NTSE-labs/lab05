"""Unit-тесты для операций аннотации, метаданных и фильтрации."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from dataset_service import (
    DatasetAnalysis,
    calculate_color_histogram,
    class_counts,
    copy_dataset_with_class_names,
    copy_dataset_with_random_names,
    create_analysis_dataframe,
    create_annotation_dataframe,
    create_source_manifest,
    filter_by_class,
    filter_by_class_and_dimensions,
    get_image_properties,
    describe_dimensions,
    pixel_statistics,
    make_class_balance_plot,
    make_color_histogram_plot,
)


def test_create_annotation_has_absolute_paths(sample_dataset: Path) -> None:
    dataframe = create_annotation_dataframe(sample_dataset)
    assert list(dataframe.columns) == ["class_name", "absolute_path"]
    assert len(dataframe) == 3
    assert all(Path(path).is_absolute() for path in dataframe["absolute_path"])


def test_create_analysis_metadata_and_labels(sample_dataset: Path) -> None:
    dataframe = create_analysis_dataframe(sample_dataset)
    assert set(dataframe["class_label"]) == {0, 1}
    assert set(dataframe["channels"]) == {3}
    assert list(dataframe["pixel_count"]) == [120, 500, 240]
    counts = class_counts(dataframe)
    assert counts.to_dict() == {"cat": 2, "dog": 1}


def test_filter_by_class_accepts_name_or_numeric_label(sample_dataset: Path) -> None:
    dataframe = create_analysis_dataframe(sample_dataset)
    assert len(filter_by_class(dataframe, "cat")) == 2
    assert len(filter_by_class(dataframe, 1)) == 1
    with pytest.raises(ValueError):
        filter_by_class(dataframe, 2)


def test_filter_by_dimensions_respects_limits(sample_dataset: Path) -> None:
    dataframe = create_analysis_dataframe(sample_dataset)
    result = filter_by_class_and_dimensions(dataframe, "cat", max_width=20, max_height=15)
    assert len(result) == 1
    assert int(result.iloc[0]["width"]) == 12
    with pytest.raises(ValueError):
        filter_by_class_and_dimensions(dataframe, "cat", max_width=0, max_height=15)
    with pytest.raises(ValueError, match="целыми числами"):
        filter_by_class_and_dimensions(dataframe, "cat", max_width="abc", max_height=15)  # type: ignore[arg-type]


def test_manifest_has_iso_dates(sample_dataset: Path) -> None:
    dataframe = create_source_manifest(sample_dataset)
    assert "modified_at" in dataframe.columns
    assert "source_url" in dataframe.columns
    assert len(dataframe) == 3
    assert "T" in dataframe.iloc[0]["modified_at"]


def test_manifest_imports_source_url_from_lab01(sample_dataset: Path) -> None:
    manifest_path = sample_dataset.parent / "manifest.jsonl"
    manifest_path.write_text(
        '{"file": "dataset/cat/cat1.jpg", "source_url": "https://example.test/cat"}\n',
        encoding="utf-8",
    )
    dataframe = create_source_manifest(sample_dataset)
    row = dataframe.loc[dataframe["relative_path"] == "cat/cat1.jpg"].iloc[0]
    assert row["source_url"] == "https://example.test/cat"


def test_class_named_copy(sample_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "class_copy"
    annotation = tmp_path / "class_annotations.csv"
    count = copy_dataset_with_class_names(sample_dataset, output, annotation)
    assert count == 3
    assert (output / "cat_0000.jpg").is_file()
    assert (output / "cat_0001.jpg").is_file()
    assert (output / "dog_0000.jpg").is_file()
    assert len(pd.read_csv(annotation)) == 3


def test_random_copy_has_unique_names(sample_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "random_copy"
    annotation = tmp_path / "random_annotations.csv"
    count = copy_dataset_with_random_names(sample_dataset, output, annotation, seed=10)
    files = list(output.iterdir())
    assert count == 3
    assert len({path.name for path in files}) == 3
    assert all(path.name.startswith("dataset_") for path in files)
    assert len(pd.read_csv(annotation)) == 3


def test_color_histogram_has_three_256_bin_arrays(sample_dataset: Path) -> None:
    image = sample_dataset / "cat" / "cat1.jpg"
    histograms = calculate_color_histogram(image)
    assert len(histograms) == 3
    assert all(values.shape == (256,) for values in histograms)
    assert all(float(values.sum()) > 0 for values in histograms)


def test_invalid_dataset_and_image_raise_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        create_annotation_dataframe(tmp_path / "missing")
    bad_image = tmp_path / "not_an_image.jpg"
    bad_image.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="не смог прочитать"):
        get_image_properties(bad_image)


def test_analysis_is_lazy_and_reuses_metadata(sample_dataset: Path) -> None:
    analysis = DatasetAnalysis(sample_dataset)
    with patch("dataset_service.get_image_properties", wraps=get_image_properties) as read:
        assert class_counts(analysis.dataframe()).to_dict() == {"cat": 2, "dog": 1}
        assert read.call_count == 0
        first = analysis.dataframe(True)
        second = analysis.dataframe(True)
        pd.testing.assert_frame_equal(first, second)
        assert read.call_count == 3
        first.loc[0, "width"] = -1
        assert analysis.dataframe(True).loc[0, "width"] == 12


@pytest.mark.parametrize("change", ["add", "delete", "modify"])
def test_analysis_invalidates_cache_when_files_change(sample_dataset: Path, change: str) -> None:
    analysis = DatasetAnalysis(sample_dataset)
    analysis.dataframe(True)
    path = sample_dataset / "cat/cat1.jpg"
    if change == "add":
        path = sample_dataset / "dog/new.jpg"
    if change == "delete":
        path.unlink()
    else:
        assert cv2.imwrite(str(path), np.zeros((31, 43, 3), np.uint8))
    result = analysis.dataframe(True)
    pd.testing.assert_frame_equal(result, create_analysis_dataframe(sample_dataset))


def test_failed_metadata_read_can_be_retried(sample_dataset: Path) -> None:
    analysis = DatasetAnalysis(sample_dataset)
    path = sample_dataset / "cat/cat1.jpg"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError):
        analysis.dataframe(True)
    assert cv2.imwrite(str(path), np.zeros((8, 9), np.uint8))
    assert analysis.dataframe(True).loc[0, "channels"] == 1


def test_statistics_and_filters_do_not_modify_source(sample_dataset: Path) -> None:
    dataframe = create_analysis_dataframe(sample_dataset)
    original = dataframe.copy(deep=True)
    stats = pixel_statistics(dataframe)
    assert stats.loc["cat", ["min", "max", "mean", "count"]].tolist() == [120, 500, 310, 2]
    assert describe_dimensions(dataframe).loc["count", "width"] == 3
    filtered = filter_by_class_and_dimensions(dataframe, 0, 12, 10)
    assert len(filtered) == 1  # Границы включены.
    filtered.loc[:, "width"] = 999
    pd.testing.assert_frame_equal(dataframe, original)


def test_histogram_and_plots_are_repeatable(sample_dataset: Path) -> None:
    path = sample_dataset / "cat/cat1.jpg"
    for values in calculate_color_histogram(path):
        assert values.sum() == 120
        assert values[40] == 120
    first = make_color_histogram_plot(path)
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert make_color_histogram_plot(path) == first
    assert make_class_balance_plot(create_annotation_dataframe(sample_dataset)).startswith(b"\x89PNG")


@pytest.mark.parametrize("copy_function", [copy_dataset_with_class_names, copy_dataset_with_random_names])
@pytest.mark.parametrize("nested", [False, True])
def test_copy_rejects_source_and_nested_destinations(copy_function, nested: bool, sample_dataset: Path) -> None:
    destination = sample_dataset / "copy" if nested else sample_dataset
    with pytest.raises(ValueError, match="внутри исходного"):
        copy_function(sample_dataset, destination)


def test_annotation_rejects_unknown_class(sample_dataset: Path) -> None:
    other = sample_dataset / "bird"
    other.mkdir()
    (other / "bird.jpg").write_bytes(b"image")
    with pytest.raises(ValueError, match="bird"):
        create_annotation_dataframe(sample_dataset)
