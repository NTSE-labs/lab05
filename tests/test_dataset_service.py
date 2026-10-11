"""Unit-тесты для операций аннотации, метаданных и фильтрации."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from dataset_service import (
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
)


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> Path:
    """Создаёт временный датасет с двумя классами и изображениями разных размеров."""
    root = tmp_path / "dataset"
    (root / "cat").mkdir(parents=True)
    (root / "dog").mkdir(parents=True)
    assert cv2.imwrite(str(root / "cat" / "cat1.jpg"), np.full((10, 12, 3), 40, np.uint8))
    assert cv2.imwrite(str(root / "cat" / "cat2.jpg"), np.full((20, 25, 3), 80, np.uint8))
    assert cv2.imwrite(str(root / "dog" / "dog1.jpg"), np.full((15, 16, 3), 120, np.uint8))
    return root


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
