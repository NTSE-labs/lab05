"""GUI приложения для сбора, подготовки и анализа датасета (Flet)."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import sys

import flet as ft

from dataset_service import (
    DatasetAnalysis,
    class_counts,
    copy_dataset_with_class_names,
    copy_dataset_with_random_names,
    create_source_manifest,
    filter_by_class,
    filter_by_class_and_dimensions,
    describe_dimensions,
    pixel_statistics,
    make_class_balance_plot,
    make_color_histogram_plot,
    save_dataframe,
)
from iterators import ClassIterator
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
    analysis_session = DatasetAnalysis(dataset_dir) if dataset_dir else None
    analysis_busy = False
    viewer_iterators = {}
    viewer_positions = {}
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
    analysis_text = ft.Text("Выберите действие анализа.", selectable=True)
    histogram_number = ft.TextField(label="Номер изображения для гистограммы", value="1", width=300)
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
        nonlocal dataset_dir, analysis_session
        if analysis_busy:
            return
        try:
            selected = await source_picker.get_directory_path(
                dialog_title="Выберите корневую папку dataset (внутри cat и dog)"
            )
            if selected is None:
                return
            if analysis_busy:
                return
            candidate = Path(selected).expanduser().resolve()
            if not candidate.is_dir():
                raise ValueError("Выбранный путь не является папкой.")
            if not any((candidate / label).is_dir() for label in ("cat", "dog")):
                raise ValueError("В выбранной папке должны быть подпапки cat и/или dog.")
            dataset_dir = candidate
            analysis_session = DatasetAnalysis(dataset_dir)
            analysis_text.value = "Выберите действие анализа."
            chart_preview.visible = False
            image_preview.visible = False
            image_path_text.value = "Изображение ещё не выбрано."
            viewer_annotations.update(original=None, class_named=None, random=None)
            viewer_selector.value = "original"
            viewer_iterators.clear()
            viewer_positions.clear()
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
            viewer_iterators.clear()
            viewer_positions.clear()
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
            key = (viewer_key, class_name)
            if key not in viewer_iterators:
                viewer_iterators[key] = ClassIterator(class_name, dataset_dir, annotation_path)
                viewer_positions[key] = 0
            path = next(viewer_iterators[key], None)
            if path is None:
                viewer_iterators[key] = ClassIterator(class_name, dataset_dir, annotation_path)
                viewer_positions[key] = 0
                path = next(viewer_iterators[key], None)
            if path is None:
                raise ValueError(f"В выбранном наборе нет изображений класса {class_name}.")
            viewer_positions[key] += 1
            if not path.is_file():
                raise FileNotFoundError(f"Изображение не найдено: {path}")
            content = path.read_bytes()
            if not content:
                raise ValueError(f"Файл пуст: {path}")
            image_preview.src = base64.b64encode(content).decode("ascii")
            image_preview.visible = True
            image_path_text.value = str(path)
            set_status(
                f"Изображение {viewer_positions[key]} класса {class_name}; "
                f"набор: {viewer_key}."
            )
        except (OSError, ValueError) as error:
            set_status(f"Не удалось открыть изображение: {error}", True)
        except Exception as error:
            set_status(f"Ошибка просмотра изображения: {error}", True)
        page.update()

    def perform_analysis(action: str, session: DatasetAnalysis, selected_class: str,
                         width_text: str, height_text: str, image_number: str):
        """Вычисления в рабочем потоке без обращения к контролам."""
        root = session.dataset_dir.parent
        if action == "histogram":
            number = int(image_number.strip())
            rows = filter_by_class(session.dataframe(), selected_class)
            paths = sorted(Path(value) for value in rows["absolute_path"])
            if not 1 <= number <= len(paths):
                raise ValueError(f"Номер изображения должен быть от 1 до {len(paths)} для {selected_class}.")
            path = paths[number - 1]
            return f"Гистограмма изображения: {path}", make_color_histogram_plot(path), "Гистограмма построена."
        if action == "filter_dimensions":
            width, height = int(width_text.strip()), int(height_text.strip())
            if width <= 0 or height <= 0:
                raise ValueError("Максимальная ширина и высота должны быть больше нуля.")
            result = filter_by_class_and_dimensions(session.dataframe(True), selected_class, width, height)
            if result.empty:
                message = (
                    f"Изображения класса {selected_class} с размерами не более "
                    f"{width}×{height} не найдены. CSV не создан."
                )
                return message, None, message
            output = root / f"filtered_{selected_class}_{width}x{height}.csv"
        elif action == "filter_class":
            result = filter_by_class(session.dataframe(), selected_class)
            output = root / f"filtered_{selected_class}.csv"
        elif action == "export":
            result = session.dataframe(True)
            output = root / "analysis_dataframe.csv"
        elif action == "counts":
            counts = class_counts(session.dataframe())
            balance = "Классы сбалансированы." if counts.nunique() == 1 else "Классы не сбалансированы."
            text = f"Всего изображений: {int(counts.sum())}\n{counts.to_string()}\n{balance}"
            return text, None, "Количество изображений подсчитано."
        elif action == "balance":
            return "Количество изображений по классам", make_class_balance_plot(session.dataframe()), "Диаграмма построена."
        elif action == "dimensions":
            return describe_dimensions(session.dataframe(True)).round(2).to_string(), None, "Статистика размеров рассчитана."
        elif action == "pixels":
            return pixel_statistics(session.dataframe(True)).round(2).to_string(), None, "Статистика пикселей рассчитана."
        else:
            raise ValueError(f"Неизвестное действие: {action}")
        save_dataframe(result, output)
        return f"Строк: {len(result)}. CSV: {output}", None, "CSV сохранён."

    async def run_analysis(action: str) -> None:
        nonlocal analysis_busy
        if analysis_busy:
            return
        if analysis_session is None:
            set_status("Сначала выберите датасет.", True)
            return
        analysis_busy = True
        for button in analysis_buttons:
            button.disabled = True
        try:
            set_status("Выполняется выбранное действие…")
            text, chart, message = await asyncio.to_thread(
                perform_analysis, action, analysis_session, class_selector.value or "cat",
                max_width_field.value or "", max_height_field.value or "", histogram_number.value or "",
            )
            analysis_text.value = text
            chart_preview.visible = chart is not None
            if chart is not None:
                chart_preview.src = base64.b64encode(chart).decode("ascii")
            set_status(message)
        except (OSError, ValueError, KeyError, TypeError) as error:
            set_status(f"Не удалось выполнить действие: {error}", True)
        except Exception as error:
            set_status(f"Ошибка анализа: {error}", True)
        finally:
            analysis_busy = False
            for button in analysis_buttons:
                button.disabled = False
            page.update()

    analysis_buttons = []

    def analysis_button(title: str, action: str) -> ft.Button:
        async def on_click(event: ft.ControlEvent) -> None:
            await run_analysis(action)
        button = ft.Button(content=title, on_click=on_click)
        analysis_buttons.append(button)
        return button

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
            analysis_button("Количество и баланс классов", "counts"),
            analysis_button("Статистика размеров", "dimensions"),
            analysis_button("Статистика пикселей по классам", "pixels"),
            analysis_button("Диаграмма баланса", "balance"),
            analysis_button("Сохранить таблицу анализа CSV", "export"),
        ], wrap=True),
        ft.Text("Размеры читаются при первом запросе; при изменении файлов пересчитываются."),
        ft.Row([class_selector, analysis_button("Фильтр по классу → CSV", "filter_class")], wrap=True),
        ft.Row([max_width_field, max_height_field,
                analysis_button("Фильтр по классу и размерам → CSV", "filter_dimensions")], wrap=True),
        ft.Text("Гистограмма одного изображения: номер в отсортированном списке выбранного класса."),
        ft.Row([histogram_number, analysis_button("Построить цветовую гистограмму", "histogram")], wrap=True),
        analysis_text,
        chart_preview,
        ft.Divider(),
        status,
    )


if __name__ == "__main__":
    ft.run(main)
