from pathlib import Path

import cv2
import numpy as np
import pytest


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


