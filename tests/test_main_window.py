"""Проверки обработчиков настоящих Flet-контролов без запуска окна."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import flet as ft
import pandas as pd
import pytest

import main_window
from dataset_service import get_image_properties


@pytest.fixture()
def ui(monkeypatch, sample_dataset: Path):
    controls = []
    page = SimpleNamespace(window=SimpleNamespace(), services=[],
                           update=lambda: None, add=lambda *items: controls.extend(items))
    monkeypatch.setattr(main_window, "DEFAULT_DATASET", sample_dataset)
    main_window.main(page)
    flat = []
    for control in controls:
        flat.append(control)
        flat.extend(getattr(control, "controls", []))
    buttons = {control.content: control for control in flat if isinstance(control, ft.Button)}
    fields = {control.label: control for control in flat if isinstance(control, ft.TextField)}
    images = [control for control in flat if isinstance(control, ft.Image)]
    return SimpleNamespace(page=page, controls=controls, buttons=buttons,
                           fields=fields, images=images, status=controls[-1])


def click(ui, title: str) -> None:
    asyncio.run(ui.buttons[title].on_click(None))


def test_counts_and_class_filter_work_without_reading_images(ui, sample_dataset: Path) -> None:
    with patch("dataset_service.get_image_properties", side_effect=AssertionError("unexpected read")):
        click(ui, "Количество и баланс классов")
        assert ui.status.value == "Количество изображений подсчитано."
        assert not (sample_dataset.parent / "analysis_dataframe.csv").exists()
        click(ui, "Фильтр по классу → CSV")
    assert len(pd.read_csv(sample_dataset.parent / "filtered_cat.csv")) == 2
    assert not (sample_dataset.parent / "source_files.csv").exists()


def test_dimension_actions_share_cached_metadata(ui, sample_dataset: Path) -> None:
    with patch("dataset_service.get_image_properties", wraps=get_image_properties) as read:
        click(ui, "Статистика размеров")
        click(ui, "Статистика пикселей по классам")
        click(ui, "Сохранить таблицу анализа CSV")
        assert read.call_count == 3
    assert len(pd.read_csv(sample_dataset.parent / "analysis_dataframe.csv")) == 3
    assert ui.status.value == "CSV сохранён."


@pytest.mark.parametrize("existing_csv", [False, True])
def test_empty_dimension_filter_skips_csv(ui, sample_dataset: Path, existing_csv: bool) -> None:
    output = sample_dataset.parent / "filtered_cat_1x1.csv"
    if existing_csv:
        output.write_bytes(b"previous result")
    ui.fields["Макс. ширина"].value = "1"
    ui.fields["Макс. высота"].value = "1"
    click(ui, "Фильтр по классу и размерам → CSV")
    assert "не найдены. CSV не создан." in ui.status.value
    if existing_csv:
        assert output.read_bytes() == b"previous result"
    else:
        assert not output.exists()


def test_nonempty_dimension_filter_saves_csv(ui, sample_dataset: Path) -> None:
    ui.fields["Макс. ширина"].value = "12"
    ui.fields["Макс. высота"].value = "10"
    click(ui, "Фильтр по классу и размерам → CSV")
    result = pd.read_csv(sample_dataset.parent / "filtered_cat_12x10.csv")
    assert len(result) == 1
    assert Path(result.iloc[0]["absolute_path"]).name == "cat1.jpg"
    assert ui.status.value == "CSV сохранён."


def test_histogram_is_repeatable_and_independent_from_viewer(ui) -> None:
    title = "Построить цветовую гистограмму"
    click(ui, title)
    first = ui.images[1].src
    ui.buttons["Следующая кошка"].on_click(None)
    click(ui, title)
    assert ui.images[1].src == first
    ui.buttons["Следующая кошка"].on_click(None)
    assert "cat2.jpg" in ui.status.value or "Изображение 2" in ui.status.value
    ui.fields["Номер изображения для гистограммы"].value = "2"
    click(ui, title)
    assert ui.images[1].src != first
    assert ui.status.value == "Гистограмма построена."


@pytest.mark.parametrize("number", ["0", "3", "abc", ""])
def test_invalid_histogram_selection_restores_buttons(ui, number: str) -> None:
    ui.fields["Номер изображения для гистограммы"].value = number
    click(ui, "Построить цветовую гистограмму")
    assert ui.status.value.startswith("Не удалось выполнить действие:")
    assert all(not button.disabled for button in ui.buttons.values())


def test_analysis_rejects_duplicate_clicks_while_busy(ui, monkeypatch) -> None:
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def controlled_worker(function, *args):
            calls.append(args[0])
            started.set()
            await release.wait()
            return function(*args)

        monkeypatch.setattr(main_window.asyncio, "to_thread", controlled_worker)
        button = ui.buttons["Количество и баланс классов"]
        task = asyncio.create_task(button.on_click(None))
        await started.wait()
        assert button.disabled
        try:
            await button.on_click(None)
            assert calls == ["counts"]
        finally:
            release.set()
            await task
        assert not button.disabled

    asyncio.run(scenario())


def test_analysis_without_dataset_shows_message(monkeypatch, tmp_path: Path) -> None:
    controls = []
    page = SimpleNamespace(window=SimpleNamespace(), services=[],
                           update=lambda: None, add=lambda *items: controls.extend(items))
    monkeypatch.setattr(main_window, "DEFAULT_DATASET", tmp_path / "missing")
    main_window.main(page)
    for row in controls:
        for control in getattr(row, "controls", []):
            if isinstance(control, ft.Button) and control.content == "Статистика размеров":
                asyncio.run(control.on_click(None))
    assert controls[-1].value == "Сначала выберите датасет."
