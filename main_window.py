"""GUI приложения для сбора, подготовки и анализа датасета (Flet)."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import sys

import pandas as pd
import flet as ft

from dataset_service import (
    calculate_color_histogram,
    class_counts,
    copy_dataset_with_class_names,
    copy_dataset_with_random_names,
    create_analysis_dataframe,
    create_source_manifest,
    filter_by_class,
    filter_by_class_and_dimensions,
    get_class_balance_text,
    make_class_balance_plot,
    make_color_histogram_plot,
    save_dataframe,
)
from scraper_runner import find_lab01_script

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR.parent / "lab01" / "dataset"


def main(page: ft.Page) -> None:
    """Создаёт GUI с функциями лабораторных 1–4."""
    page.title = "Работа с датасетом: сбор, обработка и анализ"
    page.window.width = 1180
    page.window.height = 900
    page.padding = 18
    page.scroll = ft.ScrollMode.AUTO

    dataset_dir: Path | None = DEFAULT_DATASET if DEFAULT_DATASET.is_dir() else None
    analysis_df = None
    next_index = {"cat": 0, "dog": 0}
    viewer_annotations: dict[str, Path | None] = {
        "original": None,
        "class_named": None,
        "random": None,
    }
    scraper_process = None

    dataset_label = ft.Text(
        f"Выбран датасет: {dataset_dir}" if dataset_dir else "Датасет не выбран.",
        selectable=True,
    )
    status = ft.Text("Готово.")
    scraper_log = ft.TextField(
        label="Вывод веб-скрапера lab01", multiline=True, min_lines=5, max_lines=12,
        read_only=True, expand=True,
    )
    image_path_text = ft.Text("Изображение ещё не выбрано.", selectable=True)
    image_preview = ft.Image(src="", visible=False, width=500, height=330,
                             fit=ft.BoxFit.CONTAIN)
    analysis_text = ft.Text("Нажмите «Выполнить анализ датасета».", selectable=True)
    chart_preview = ft.Image(src="", visible=False, width=700, height=370,
                             fit=ft.BoxFit.CONTAIN)
    class_selector = ft.Dropdown(
        label="Класс для анализа", value="cat", width=180,
        options=[ft.dropdown.Option("cat", "Кошки (cat)"), ft.dropdown.Option("dog", "Собаки (dog)")],
    )
    max_width_field = ft.TextField(label="Макс. ширина", value="1000", width=150)
    max_height_field = ft.TextField(label="Макс. высота", value="1000", width=150)
    viewer_selector = ft.Dropdown(
        label="Набор для просмотра", value="original", width=230,
        options=[
            ft.dropdown.Option("original", "Исходный датасет"),
            ft.dropdown.Option("class_named", "Копия: класс_номер"),
            ft.dropdown.Option("random", "Копия: случайный номер"),
        ],
    )

    source_picker = ft.FilePicker()
    destination_picker = ft.FilePicker()
    page.services.extend([source_picker, destination_picker])

    def set_status(message: str, is_error: bool = False) -> None:
        status.value = message
        status.color = ft.Colors.RED if is_error else ft.Colors.GREEN
        page.update()

    async def choose_dataset(event: ft.ControlEvent) -> None:
        nonlocal dataset_dir, analysis_df
        try:
            selected = await source_picker.get_directory_path(
                dialog_title="Выберите корневую папку dataset (внутри cat и dog)"
            )
            if selected is None:
                return
            candidate = Path(selected).expanduser().resolve()
            if not candidate.is_dir():
                raise ValueError("Выбранный путь не является папкой.")
            if not any((candidate / label).is_dir() for label in ("cat", "dog")):
                raise ValueError("В выбранной папке должны быть подпапки cat и/или dog.")
            dataset_dir = candidate
            analysis_df = None
            viewer_annotations.update(original=None, class_named=None, random=None)
            viewer_selector.value = "original"
            next_index.update(cat=0, dog=0)
            dataset_label.value = f"Выбран датасет: {dataset_dir}"
            set_status("Датасет выбран.")
        except (OSError, ValueError) as error:
            set_status(f"Не удалось выбрать датасет: {error}", True)
        except Exception as error:
            set_status(f"Ошибка при выборе датасета: {error}", True)

    async def run_scraper(event: ft.ControlEvent) -> None:
        """Запускает настоящий lab01/main.py, читает вывод асинхронно."""
        nonlocal scraper_process
        try:
            script = find_lab01_script(PROJECT_DIR)
            scraper_log.value = f"Запуск: {script}\n"
            page.update()
            scraper_process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", str(script),
                cwd=str(script.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert scraper_process.stdout is not None
            while True:
                line = await scraper_process.stdout.readline()
                if not line:
                    break
                scraper_log.value = ((scraper_log.value or "") + line.decode("utf-8", errors="replace"))[-30000:]
                page.update()
            exit_code = await scraper_process.wait()
            scraper_log.value = (scraper_log.value or "") + f"\nПроцесс завершён, код: {exit_code}\n"
            if exit_code == 0:
                set_status("Веб-скрапинг завершён. При необходимости выберите обновлённый датасет.")
            else:
                set_status(f"Скрапер завершился с кодом {exit_code}; смотрите лог.", True)
        except FileNotFoundError as error:
            set_status(str(error), True)
        except OSError as error:
            set_status(f"Не удалось запустить скрапер: {error}", True)
        except Exception as error:
            set_status(f"Ошибка веб-скрапинга: {error}", True)
        finally:
            scraper_process = None

    async def stop_scraper(event: ft.ControlEvent) -> None:
        nonlocal scraper_process
        if scraper_process is None or scraper_process.returncode is not None:
            set_status("Активный процесс скрапера не найден.", True)
            return
        scraper_process.terminate()
        set_status("Отправлен запрос на остановку скрапера.")

    async def export_manifest(event: ft.ControlEvent) -> None:
        if dataset_dir is None:
            set_status("Сначала выберите датасет.", True)
            return
        try:
            manifest = create_source_manifest(dataset_dir)
            target = dataset_dir.parent / "source_files.csv"
            save_dataframe(manifest, target)
            set_status(f"Манифест сохранён: {target} (строк: {len(manifest)}).")
        except (OSError, ValueError) as error:
            set_status(f"Не удалось создать манифест: {error}", True)

    async def make_copy(kind: str) -> None:
        if dataset_dir is None:
            set_status("Сначала выберите датасет.", True)
            return
        try:
            destination_text = await destination_picker.get_directory_path(
                dialog_title="Выберите папку, в которой создать копию"
            )
            if destination_text is None:
                return
            root = Path(destination_text).expanduser().resolve()
            if kind == "class":
                output = root / "class_named_dataset"
                annotations = root / "class_named_annotations.csv"
                count = copy_dataset_with_class_names(dataset_dir, output, annotations)
                viewer_annotations["class_named"] = annotations.resolve()
                viewer_selector.value = "class_named"
            else:
                output = root / "random_named_dataset"
                annotations = root / "random_annotations.csv"
                count = copy_dataset_with_random_names(dataset_dir, output, annotations)
                viewer_annotations["random"] = annotations.resolve()
                viewer_selector.value = "random"
            next_index.update(cat=0, dog=0)
            set_status(f"Создано файлов: {count}. Копия: {output}. CSV: {annotations}")
        except (OSError, ValueError) as error:
            set_status(f"Не удалось создать копию: {error}", True)
        except Exception as error:
            set_status(f"Ошибка при копировании: {error}", True)

    async def make_class_copy(event: ft.ControlEvent) -> None:
        await make_copy("class")

    async def make_random_copy(event: ft.ControlEvent) -> None:
        await make_copy("random")

    def show_next(class_name: str) -> None:
        if dataset_dir is None:
            set_status("Сначала выберите датасет.", True)
            return
        try:
            viewer_key = viewer_selector.value or "original"
            annotation_path = viewer_annotations.get(viewer_key)
            if viewer_key != "original":
                if annotation_path is None or not annotation_path.is_file():
                    raise ValueError(
                        "Сначала создайте выбранную копию датасета: "
                        "для неё пока нет CSV-аннотации."
                    )
                annotation = pd.read_csv(annotation_path)
                paths = sorted(
                    Path(value).expanduser().resolve()
                    for value in annotation.loc[
                        annotation["class_name"] == class_name, "absolute_path"
                    ].dropna().astype(str)
                )
            else:
                paths = sorted(
                    p.resolve() for p in (dataset_dir / class_name).glob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                )
            if not paths:
                set_status(f"В выбранном наборе нет изображений класса {class_name}.", True)
                return
            index = next_index[class_name] % len(paths)
            path = paths[index]
            next_index[class_name] = index + 1
            if not path.is_file():
                raise FileNotFoundError(f"Изображение не найдено: {path}")
            content = path.read_bytes()
            if not content:
                raise ValueError(f"Файл пуст: {path}")
            image_preview.src = base64.b64encode(content).decode("ascii")
            image_preview.visible = True
            image_path_text.value = str(path)
            set_status(
                f"Изображение {index + 1} из {len(paths)} класса {class_name}; "
                f"набор: {viewer_key}."
            )
        except (OSError, ValueError) as error:
            set_status(f"Не удалось открыть изображение: {error}", True)
        except Exception as error:
            set_status(f"Ошибка просмотра изображения: {error}", True)
        page.update()

    async def analyze_dataset(event: ft.ControlEvent) -> None:
        nonlocal analysis_df
        if dataset_dir is None:
            set_status("Сначала выберите датасет.", True)
            return
        try:
            analysis_df = create_analysis_dataframe(dataset_dir)
            analysis_file = dataset_dir.parent / "analysis_dataframe.csv"
            save_dataframe(analysis_df, analysis_file)
            source_manifest = create_source_manifest(dataset_dir)
            save_dataframe(source_manifest, dataset_dir.parent / "source_files.csv")
            analysis_text.value = get_class_balance_text(analysis_df)
            chart_preview.src = base64.b64encode(make_class_balance_plot(analysis_df)).decode("ascii")
            chart_preview.visible = True
            set_status(
                f"Анализ завершён: {len(analysis_df)} изображений. "
                f"CSV: {analysis_file}"
            )
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            set_status(f"Не удалось выполнить анализ: {error}", True)
        except Exception as error:
            set_status(f"Ошибка анализа: {error}", True)
        page.update()

    async def filter_current_class(event: ft.ControlEvent) -> None:
        if analysis_df is None:
            set_status("Сначала выполните анализ датасета.", True)
            return
        try:
            selected_class = class_selector.value or "cat"
            result = filter_by_class(analysis_df, selected_class)
            output = dataset_dir.parent / f"filtered_{selected_class}.csv"
            save_dataframe(result, output)
            set_status(f"Для класса {selected_class} найдено {len(result)} строк. CSV: {output}")
        except (OSError, ValueError, TypeError) as error:
            set_status(f"Не удалось отфильтровать данные: {error}", True)

    async def filter_dimensions(event: ft.ControlEvent) -> None:
        if analysis_df is None or dataset_dir is None:
            set_status("Сначала выполните анализ датасета.", True)
            return
        try:
            # Не превращаем некорректный ввод в ноль: показываем понятную ошибку.
            max_width = int((max_width_field.value or "").strip())
            max_height = int((max_height_field.value or "").strip())
            selected_class = class_selector.value or "cat"
            result = filter_by_class_and_dimensions(
                analysis_df, selected_class, max_width, max_height
            )
            output = dataset_dir.parent / f"filtered_{selected_class}_{max_width}x{max_height}.csv"
            save_dataframe(result, output)
            set_status(f"После фильтрации: {len(result)} изображений. CSV: {output}")
        except (ValueError, TypeError) as error:
            set_status(f"Проверьте введённые размеры: {error}", True)
        except OSError as error:
            set_status(f"Не удалось сохранить результат фильтрации: {error}", True)

    async def show_color_histogram(event: ft.ControlEvent) -> None:
        if dataset_dir is None:
            set_status("Сначала выберите датасет.", True)
            return
        try:
            selected_class = class_selector.value or "cat"
            class_dir = dataset_dir / selected_class
            paths = sorted(
                p for p in class_dir.glob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            if not paths:
                raise ValueError(f"В классе {selected_class} нет изображений.")
            # Последовательно выбираем очередное изображение, чтобы результат менялся.
            path = paths[next_index[selected_class] % len(paths)]
            next_index[selected_class] += 1
            histogram = calculate_color_histogram(path)
            if not histogram or any(values.size != 256 for values in histogram):
                raise ValueError("Гистограмма имеет неожиданный формат.")
            chart_preview.src = base64.b64encode(make_color_histogram_plot(path)).decode("ascii")
            chart_preview.visible = True
            set_status(f"Показана цветовая гистограмма для: {path.name}")
        except (OSError, ValueError) as error:
            set_status(f"Не удалось построить гистограмму: {error}", True)
        except Exception as error:
            set_status(f"Ошибка построения гистограммы: {error}", True)
        page.update()

    page.add(
        ft.Text("Обработка и анализ датасета кошек и собак", size=27, weight=ft.FontWeight.BOLD),
        ft.Text("Сбор изображений (лаба 1), подготовка и копирование (лабы 2–3), анализ (лаба 4)."),
        ft.Divider(),
        ft.Text("1. Веб-скрапинг", size=21, weight=ft.FontWeight.BOLD),
        ft.Row([
            ft.Button(content="Запустить скрапер lab01", on_click=run_scraper),
            ft.Button(content="Остановить скрапер", on_click=stop_scraper),
        ], wrap=True),
        scraper_log,
        ft.Divider(),
        ft.Text("2. Датасет и копии", size=21, weight=ft.FontWeight.BOLD),
        ft.Row([
            ft.Button(content="Выбрать датасет", on_click=choose_dataset),
            ft.Button(content="Создать манифест CSV", on_click=export_manifest),
            ft.Button(content="Копия: класс_номер", on_click=make_class_copy),
            ft.Button(content="Копия: случайный номер", on_click=make_random_copy),
        ], wrap=True),
        dataset_label,
        ft.Row([
            viewer_selector,
            ft.Button(content="Следующая кошка", on_click=lambda event: show_next("cat")),
            ft.Button(content="Следующая собака", on_click=lambda event: show_next("dog")),
        ], wrap=True),
        image_preview,
        image_path_text,
        ft.Divider(),
        ft.Text("3. Исследовательский анализ (лаба 4)", size=21, weight=ft.FontWeight.BOLD),
        ft.Row([
            ft.Button(content="Выполнить анализ датасета", on_click=analyze_dataset),
            ft.Button(content="Фильтр по классу", on_click=filter_current_class),
            ft.Button(content="Построить цветовую гистограмму", on_click=show_color_histogram),
        ], wrap=True),
        ft.Row([class_selector, max_width_field, max_height_field,
                ft.Button(content="Фильтр по классу и размерам", on_click=filter_dimensions)], wrap=True),
        analysis_text,
        chart_preview,
        ft.Divider(),
        status,
    )


if __name__ == "__main__":
    ft.run(main)
