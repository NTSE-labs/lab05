"""Контракт итераторов и независимость последовательностей просмотра."""

from pathlib import Path

import pandas as pd
import pytest

from dataset_service import copy_dataset_with_class_names, copy_dataset_with_random_names
from iterators import ClassIterator, DatasetIterator


def test_dataset_iterator_order_and_exhaustion(sample_dataset: Path) -> None:
    iterator = DatasetIterator(sample_dataset)
    assert iter(iterator) is iterator
    assert next(iterator) == ("cat", sample_dataset / "cat/cat1.jpg")
    assert list(iterator) == [
        ("cat", sample_dataset / "cat/cat2.jpg"),
        ("dog", sample_dataset / "dog/dog1.jpg"),
    ]
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(iterator)


def test_class_iterators_are_independent(sample_dataset: Path) -> None:
    cats = ClassIterator("cat", sample_dataset)
    other_cats = ClassIterator(0, sample_dataset)
    dogs = ClassIterator("dog", sample_dataset)
    assert iter(cats) is cats
    assert next(cats).name == "cat1.jpg"
    assert next(dogs).name == "dog1.jpg"
    assert next(other_cats).name == "cat1.jpg"
    assert next(cats).name == "cat2.jpg"
    assert next(cats, None) is None
    assert next(cats, None) is None
    assert len(list(ClassIterator("cat", sample_dataset))) == 2


def test_iterators_include_nested_jpegs_and_ignore_other_files(sample_dataset: Path) -> None:
    nested = sample_dataset / "cat/nested"
    nested.mkdir()
    image = nested / "photo.JPEG"
    image.write_bytes((sample_dataset / "cat/cat1.jpg").read_bytes())
    (nested / "notes.txt").write_text("ignored", encoding="utf-8")
    assert list(ClassIterator("cat", sample_dataset))[-1] == image
    assert ("cat", image) in list(DatasetIterator(sample_dataset))


def test_empty_class_and_invalid_class(sample_dataset: Path) -> None:
    for path in (sample_dataset / "dog").iterdir():
        path.unlink()
    assert list(ClassIterator("dog", sample_dataset)) == []
    with pytest.raises(ValueError):
        ClassIterator("bird", sample_dataset)


@pytest.mark.parametrize("iterator_type", [DatasetIterator, lambda root: ClassIterator("cat", root)])
def test_missing_and_empty_dataset(iterator_type, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        iterator_type(tmp_path / "missing")
    with pytest.raises(ValueError):
        iterator_type(tmp_path)


@pytest.mark.parametrize("copy_function", [copy_dataset_with_class_names, copy_dataset_with_random_names])
def test_class_iterator_uses_copy_annotation(copy_function, sample_dataset: Path, tmp_path: Path) -> None:
    output, annotation = tmp_path / "copy", tmp_path / "annotation.csv"
    copy_function(sample_dataset, output, annotation)
    rows = pd.read_csv(annotation)
    # Повторная строка CSV не должна повторять изображение в итераторе.
    pd.concat([rows, rows.iloc[:1]]).to_csv(annotation, index=False)
    paths = list(ClassIterator("cat", sample_dataset, annotation))
    assert len(paths) == 2
    assert all(path.parent == output and path.is_file() for path in paths)
    assert paths == sorted(paths)
