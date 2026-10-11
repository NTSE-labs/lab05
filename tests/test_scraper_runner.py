"""Unit-тесты поиска исполняемого скрипта скрапера."""

from pathlib import Path

import pytest

from scraper_runner import find_lab01_script


def test_find_lab01_script_in_sibling_repository(tmp_path: Path) -> None:
    root = tmp_path / "NTSE"
    lab01 = root / "lab01"
    lab05 = root / "lab05"
    lab01.mkdir(parents=True)
    lab05.mkdir()
    script = lab01 / "main.py"
    script.write_text("print('hello')", encoding="utf-8")
    assert find_lab01_script(lab05) == script.resolve()


def test_missing_lab01_script_has_actionable_error(tmp_path: Path) -> None:
    project = tmp_path / "lab05"
    project.mkdir()
    with pytest.raises(FileNotFoundError, match="Не найден скрипт"):
        find_lab01_script(project)
