"""Конечные итераторы исходного датасета и его CSV-копий."""

from pathlib import Path

import pandas as pd

from dataset_service import create_annotation_dataframe, filter_by_class


class DatasetIterator:
    """Обходит датасет в порядке путей, возвращая (класс, абсолютный путь)."""

    def __init__(self, dataset_dir: Path) -> None:
        dataframe = create_annotation_dataframe(dataset_dir)
        self._rows = iter(dataframe.itertuples(index=False, name=None))

    def __iter__(self) -> DatasetIterator:
        return self

    def __next__(self) -> tuple[str, Path]:
        class_name, path = next(self._rows)
        return class_name, Path(path)


class ClassIterator:
    """Обходит один класс без повторов; CSV позволяет просматривать копии."""

    def __init__(
        self, class_label: str | int, dataset_dir: Path,
        annotation_path: Path | None = None,
    ) -> None:
        dataframe = (
            create_annotation_dataframe(dataset_dir)
            if annotation_path is None else pd.read_csv(annotation_path)
        )
        selected = filter_by_class(dataframe, class_label)
        paths = {Path(path).expanduser().resolve() for path in selected["absolute_path"].dropna()}
        self._paths = iter(sorted(paths))

    def __iter__(self) -> ClassIterator:
        return self

    def __next__(self) -> Path:
        return next(self._paths)
