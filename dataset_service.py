"""Функции работы с датасетом, аннотацией и анализом изображений.

Модуль не зависит от GUI, поэтому его функции можно проверять unit-тестами.
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
CLASS_TO_LABEL = {"cat": 0, "dog": 1}


def find_image_files(dataset_dir: Path) -> list[Path]:
    """Возвращает отсортированные изображения внутри каталога датасета."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Папка датасета не найдена: {dataset_dir}")
    paths = sorted(
        path.resolve()
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"В папке не найдены изображения: {dataset_dir}")
    return paths


def get_class_name(image_path: Path, dataset_dir: Path) -> str:
    """Определяет класс изображения по имени первой папки под dataset_dir."""
    relative = Path(image_path).resolve().relative_to(Path(dataset_dir).resolve())
    if len(relative.parts) < 2:
        raise ValueError(
            f"Изображение должно находиться в отдельной папке класса: {image_path}"
        )
    return relative.parts[0].lower()


def create_annotation_dataframe(dataset_dir: Path) -> pd.DataFrame:
    """Создаёт DataFrame с названием класса и абсолютным путём к файлу."""
    paths = find_image_files(dataset_dir)
    rows = [
        {
            "class_name": get_class_name(path, dataset_dir),
            "absolute_path": str(path.resolve()),
        }
        for path in paths
    ]
    dataframe = pd.DataFrame(rows, columns=["class_name", "absolute_path"])
    unknown = sorted(set(dataframe["class_name"]) - set(CLASS_TO_LABEL))
    if unknown:
        raise ValueError(
            "Ожидаются только классы cat и dog; обнаружены: " + ", ".join(unknown)
        )
    return dataframe


def create_source_manifest(dataset_dir: Path) -> pd.DataFrame:
    """Формирует аннотацию лабораторной 2 с датой изменения в ISO 8601."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    paths = find_image_files(dataset_dir)
    source_urls: dict[str, str] = {}
    # Лаба 1 сохраняет URL изображений в JSONL-манифесте рядом с dataset.
    source_manifest = dataset_dir.parent / "manifest.jsonl"
    if source_manifest.is_file():
        try:
            with source_manifest.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    relative_file = str(record.get("file", "")).replace("\\", "/")
                    if relative_file:
                        source_urls[relative_file] = str(record.get("source_url", ""))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Не удалось прочитать манифест lab01: {error}") from error
    rows: list[dict[str, str]] = []
    for path in paths:
        modified_at = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        # В lab01 поле file задаётся относительно папки репозитория: dataset/cat/0000.jpg.
        relative_to_lab01 = path.relative_to(dataset_dir.parent).as_posix()
        rows.append(
            {
                "absolute_path": str(path.resolve()),
                "relative_path": path.relative_to(dataset_dir).as_posix(),
                "class_name": get_class_name(path, dataset_dir),
                "modified_at": modified_at.isoformat(),
                "source_url": source_urls.get(relative_to_lab01, ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["absolute_path", "relative_path", "class_name", "modified_at", "source_url"],
    )


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Сохраняет DataFrame в CSV и возвращает абсолютный путь к файлу."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def get_image_properties(image_path: Path) -> tuple[int, int, int]:
    """Возвращает высоту, ширину и число каналов; ошибку чтения не скрывает."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV не смог прочитать изображение: {image_path}")
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return int(height), int(width), int(channels)


def create_analysis_dataframe(dataset_dir: Path) -> pd.DataFrame:
    """Добавляет числовые метки и характеристики изображений для анализа."""
    dataframe = create_annotation_dataframe(dataset_dir)
    dataframe["class_label"] = dataframe["class_name"].map(CLASS_TO_LABEL).astype(int)
    properties = [get_image_properties(Path(path)) for path in dataframe["absolute_path"]]
    dataframe["height"] = [item[0] for item in properties]
    dataframe["width"] = [item[1] for item in properties]
    dataframe["channels"] = [item[2] for item in properties]
    dataframe["pixel_count"] = dataframe["height"] * dataframe["width"]
    return dataframe


def class_counts(dataframe: pd.DataFrame) -> pd.Series:
    """Возвращает количество записей по каждому из двух классов."""
    if "class_name" not in dataframe.columns:
        raise ValueError("В таблице отсутствует колонка class_name.")
    return dataframe["class_name"].value_counts().reindex(["cat", "dog"], fill_value=0)


def describe_dimensions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Возвращает describe() для ширины, высоты, каналов и числа пикселей."""
    needed = {"height", "width", "channels", "pixel_count", "class_label"}
    missing = needed - set(dataframe.columns)
    if missing:
        raise ValueError("Для статистики не хватает колонок: " + ", ".join(sorted(missing)))
    return dataframe[["height", "width", "channels", "pixel_count", "class_label"]].describe()


def filter_by_class(dataframe: pd.DataFrame, class_label: str | int) -> pd.DataFrame:
    """Фильтрует строки по имени класса (cat/dog) или числовой метке (0/1)."""
    if "class_name" not in dataframe.columns:
        raise ValueError("В таблице отсутствует колонка class_name.")
    if isinstance(class_label, int):
        if class_label not in (0, 1):
            raise ValueError("Числовая метка класса должна быть 0 или 1.")
        class_name = next(name for name, label in CLASS_TO_LABEL.items() if label == class_label)
    elif isinstance(class_label, str) and class_label.lower() in CLASS_TO_LABEL:
        class_name = class_label.lower()
    else:
        raise ValueError("Класс должен быть cat, dog, 0 или 1.")
    return dataframe.loc[dataframe["class_name"] == class_name].copy()


def filter_by_class_and_dimensions(
    dataframe: pd.DataFrame,
    class_label: str | int,
    max_width: int,
    max_height: int,
) -> pd.DataFrame:
    """Фильтрует изображения по классу и верхним границам ширины/высоты."""
    if isinstance(max_width, bool) or isinstance(max_height, bool):
        raise ValueError("Максимальная ширина и высота должны быть целыми числами.")
    if not isinstance(max_width, int) or not isinstance(max_height, int):
        raise ValueError("Максимальная ширина и высота должны быть целыми числами.")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("Максимальная ширина и высота должны быть больше нуля.")
    needed = {"width", "height"}
    if not needed.issubset(dataframe.columns):
        raise ValueError("В таблице отсутствуют колонки width и/или height.")
    by_class = filter_by_class(dataframe, class_label)
    return by_class.loc[
        (by_class["width"] <= max_width) & (by_class["height"] <= max_height)
    ].copy()


def pixel_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Считает минимум, максимум и среднее число пикселей по каждому классу."""
    if "pixel_count" not in dataframe.columns:
        if not {"width", "height"}.issubset(dataframe.columns):
            raise ValueError("Для расчёта pixel_count нужны width и height.")
        dataframe = dataframe.copy()
        dataframe["pixel_count"] = dataframe["width"] * dataframe["height"]
    return dataframe.groupby("class_name")["pixel_count"].agg(
        ["min", "max", "mean", "count"]
    )


def _validate_destination(source_dir: Path, destination_dir: Path) -> None:
    """Запрещает создавать копию внутри исходного датасета."""
    source = Path(source_dir).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError(
            "Папка назначения не должна находиться внутри исходного датасета. "
            "Иначе копии могут попасть в исходный набор при следующем запуске."
        )


def copy_dataset_with_class_names(
    source_dir: Path,
    destination_dir: Path,
    annotation_path: Path | None = None,
) -> int:
    """Копирует изображения как cat_0000.jpg / dog_0000.jpg и пишет CSV."""
    source_dir = Path(source_dir).expanduser().resolve()
    destination_dir = Path(destination_dir).expanduser().resolve()
    _validate_destination(source_dir, destination_dir)
    paths = find_image_files(source_dir)
    rows: list[dict[str, str]] = []
    counters = {"cat": 0, "dog": 0}
    for path in paths:
        class_name = get_class_name(path, source_dir)
        if class_name not in counters:
            raise ValueError(f"Неизвестный класс «{class_name}»: {path}")
        destination_dir.mkdir(parents=True, exist_ok=True)
        target = destination_dir / f"{class_name}_{counters[class_name]:04d}{path.suffix.lower()}"
        shutil.copy2(path, target)
        counters[class_name] += 1
        rows.append({"class_name": class_name, "absolute_path": str(target.resolve())})
    if annotation_path is None:
        annotation_path = destination_dir.parent / "class_named_annotations.csv"
    save_dataframe(pd.DataFrame(rows), annotation_path)
    return len(rows)


def copy_dataset_with_random_names(
    source_dir: Path,
    destination_dir: Path,
    annotation_path: Path | None = None,
    seed: int | None = None,
) -> int:
    """Копирует изображения под уникальными числовыми именами и пишет CSV."""
    source_dir = Path(source_dir).expanduser().resolve()
    destination_dir = Path(destination_dir).expanduser().resolve()
    _validate_destination(source_dir, destination_dir)
    paths = find_image_files(source_dir)
    rng = random.Random(seed)
    numbers = rng.sample(range(10000), k=len(paths)) if len(paths) <= 10000 else None
    if numbers is None:
        raise ValueError("В датасете больше 10000 файлов — диапазон номеров исчерпан.")
    destination_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for path, number in zip(paths, numbers, strict=True):
        class_name = get_class_name(path, source_dir)
        target = destination_dir / f"dataset_{number:04d}{path.suffix.lower()}"
        shutil.copy2(path, target)
        rows.append({"class_name": class_name, "absolute_path": str(target.resolve())})
    if annotation_path is None:
        annotation_path = destination_dir.parent / "random_annotations.csv"
    save_dataframe(pd.DataFrame(rows), annotation_path)
    return len(rows)


def calculate_color_histogram(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает три 256-биновые гистограммы изображения в каналах B, G, R."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {image_path}")
    histograms = tuple(
        cv2.calcHist([image], [channel], None, [256], [0, 256]).reshape(-1)
        for channel in range(3)
    )
    return histograms  # type: ignore[return-value]


def make_class_balance_plot(dataframe: pd.DataFrame) -> bytes:
    """Строит столбчатую диаграмму количества изображений каждого класса."""
    counts = class_counts(dataframe)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(counts.index.tolist(), counts.values.tolist())
    axis.set_title("Количество изображений по классам")
    axis.set_xlabel("Класс")
    axis.set_ylabel("Количество изображений")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _figure_to_png(figure)


def make_color_histogram_plot(image_path: Path) -> bytes:
    """Строит цветовые гистограммы BGR и возвращает PNG-изображение в памяти."""
    histograms = calculate_color_histogram(image_path)
    figure, axis = plt.subplots(figsize=(8, 4))
    for histogram, channel_name in zip(histograms, ("Blue", "Green", "Red"), strict=True):
        axis.plot(histogram, label=channel_name)
    axis.set_title(f"Гистограмма цветов: {Path(image_path).name}")
    axis.set_xlabel("Интенсивность пикселя (0–255)")
    axis.set_ylabel("Количество пикселей")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return _figure_to_png(figure)


def _figure_to_png(figure: Any) -> bytes:
    """Сохраняет matplotlib figure в PNG-байты и закрывает фигуру."""
    import io

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()


def get_class_balance_text(dataframe: pd.DataFrame) -> str:
    """Формирует краткое текстовое резюме баланса и статистики датасета."""
    counts = class_counts(dataframe)
    total = int(counts.sum())
    if total == 0:
        balance = "Нет изображений для анализа."
    elif counts.nunique() == 1:
        balance = "Классы сбалансированы по количеству изображений."
    else:
        balance = "Классы не сбалансированы по количеству изображений."
    lines = [
        f"Всего изображений: {total}",
        f"Кошки (cat): {int(counts['cat'])}",
        f"Собаки (dog): {int(counts['dog'])}",
        balance,
    ]
    stats = describe_dimensions(dataframe)
    lines.extend(["", "Статистика describe():", stats.round(2).to_string()])
    lines.extend(["", "Размеры изображений по классам:", pixel_statistics(dataframe).round(2).to_string()])
    return "\n".join(lines)
