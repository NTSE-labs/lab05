"""Поиск скрипта и запуск скрапера из лабораторной работы 1."""

from pathlib import Path
import sys
import subprocess
from typing import IO


def find_lab01_script(project_dir: Path | None = None) -> Path:
    """Находит соседний lab01/main.py относительно папки lab05."""
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent
    project_dir = Path(project_dir).expanduser().resolve()
    candidates = [
        project_dir.parent / "lab01" / "main.py",
        project_dir / "lab01" / "main.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Не найден скрипт lab01/main.py. Разместите репозитории lab01 и lab05 "
        "рядом в общей папке NTSE либо измените путь в scraper_runner.py."
    )


def start_scraper(project_dir: Path | None = None) -> subprocess.Popen[str]:
    """Запускает скрапер в отдельном процессе и возвращает объект процесса."""
    script = find_lab01_script(project_dir)
    return subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(script.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
