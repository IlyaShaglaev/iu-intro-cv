# -*- coding: utf-8 -*-
"""Автотесты занятия 10 «Гомография и масштабирование». Самопроверка студента и CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:
путь в переменной окружения `CVCOURSE_LAB_NB` (так CI ассистента гоняет решение);
`lab10_solution.ipynb` рядом — появляется в репозитории после занятия;
`lab10.ipynb` — заготовка, заполненная студентом; тесты незаполненных TODO
пропускаются, а не падают.

Данные синтетические: ни фото лекции, ни сеть тестам не нужны.
"""
import io
import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = (
    os.environ.get("CVCOURSE_LAB_NB"),
    os.path.join(HERE, "lab10_solution.ipynb"),
    os.path.join(HERE, "lab10.ipynb"),
)
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)

WANTED = (
    "compute_homography_dlt",
    "resize_cycle_test",
    "resize_composition_test",
    "consistent_downsample",
)


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab10_solution.ipynb, ни lab10.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab10", "__todo__": set()}
    exec("import cv2, numpy as np, timeit\nfrom cvcourse import metrics", g)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        hit = [w for w in WANTED if "def %s(" % w in src]
        if not hit:
            continue
        sep = "\n" * 3
        for chunk in src.split(sep):
            exec(compile(chunk, "<lab10>", "exec"), g)
            defined = [w for w in hit if "def %s(" % w in chunk]
            if defined:
                if "NotImplementedError" in chunk:
                    g["__todo__"].update(defined)
                    break
    for name in WANTED:
        if name not in g:
            pytest.skip("в ноутбуке нет функции %s" % name)
    return g


def need(sol, *names):
    """Тест пропускается, если нужный TODO ещё не заполнен."""
    todo = [n for n in names if n in sol["__todo__"]]
    if todo:
        pytest.skip("не заполнен TODO: " + ", ".join(todo))


# ------------------------------------------------------------- сцены ---

@pytest.fixture(scope="module")
def scene():
    """«Фотография» 120 × 160 с текстурой: круги, полосы, ступени."""
    yy, xx = np.mgrid[:120, :160].astype(np.float32)
    img = 110 + 50 * np.sin(xx / 19.0) * np.cos(yy / 13.0) + 0.2 * xx
    for cx, cy, r in ((40, 35, 22), (115, 80, 30), (70, 95, 14)):
        img[np.hypot(xx - cx, yy - cy) < r] += 60
    img[20:100:16, 100:150] -= 70
    return np.uint8(np.clip(img, 0, 255))


@pytest.fixture(scope="module")
def homography_scene(scene):
    """Повёрнутый и масштабированный кадр с известной гомографией."""
    h, w = scene.shape[:2]
    theta = np.radians(15)
    s = 0.85
    tx, ty = w * 0.08, h * 0.06
    H_true = np.array([
        [s * np.cos(theta), -s * np.sin(theta), tx],
        [s * np.sin(theta),  s * np.cos(theta), ty],
        [0, 0, 1]
    ], np.float64)
    rotated = cv2.warpPerspective(scene.astype(np.float64), H_true, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    dst_pts = cv2.perspectiveTransform(src_pts.reshape(1, -1, 2), H_true).reshape(-1, 2)
    return scene, rotated, H_true, src_pts, dst_pts


# ------------------------------------------------------------- тесты ---

def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_dlt_homography_shape_and_normalization(sol, homography_scene):
    """compute_homography_dlt возвращает матрицу 3×3 с H[2,2] = 1."""
    need(sol, "compute_homography_dlt")
    _, _, _, src_pts, dst_pts = homography_scene
    H = sol["compute_homography_dlt"](src_pts, dst_pts)
    assert H.shape == (3, 3), "гомография должна быть матрицей 3×3"
    assert abs(H[2, 2] - 1.0) < 1e-10, "H[2,2] должна быть нормирована к 1"


def test_dlt_homography_recovers_true(sol, homography_scene):
    """DLT восстанавливает истинную гомографию по 4 парам точек."""
    need(sol, "compute_homography_dlt")
    _, _, H_true, src_pts, dst_pts = homography_scene
    H_mine = sol["compute_homography_dlt"](src_pts, dst_pts)
    H_mine_norm = H_mine / np.linalg.norm(H_mine)
    H_true_norm = H_true / np.linalg.norm(H_true)
    d = min(np.abs(H_mine_norm - H_true_norm).max(),
            np.abs(H_mine_norm + H_true_norm).max())
    assert d < 1e-6, "DLT должна восстановить истинную гомографию (ошибка %.2e)" % d


def test_dlt_homography_matches_cv2(sol, homography_scene):
    """DLT совпадает с cv2.findHomography."""
    need(sol, "compute_homography_dlt")
    _, _, _, src_pts, dst_pts = homography_scene
    H_mine = sol["compute_homography_dlt"](src_pts, dst_pts)
    H_cv, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    d = min(np.abs(H_mine / H_mine[2, 2] - H_cv / H_cv[2, 2]).max(),
            np.abs(H_mine / H_mine[2, 2] + H_cv / H_cv[2, 2]).max())
    assert d < 1e-4, "DLT должна совпадать с cv2.findHomography (ошибка %.2e)" % d


def test_dlt_homography_applies_correctly(sol, homography_scene):
    """Гомография, восстановленная через DLT, правильно трансформирует точки."""
    need(sol, "compute_homography_dlt")
    _, _, _, src_pts, dst_pts = homography_scene
    H_mine = sol["compute_homography_dlt"](src_pts, dst_pts)
    # Применяем гомографию к исходным точкам и сравниваем с целевыми
    src_h = np.column_stack([src_pts, np.ones(len(src_pts))])
    transformed = (H_mine @ src_h.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    err = np.abs(transformed - dst_pts).max()
    assert err < 1e-4, "гомография должна правильно трансформировать точки (ошибка %.2e)" % err


def test_resize_cycle_test_positive_error(sol, scene):
    """Ошибка цикла ×2 → ×1/2 должна быть > 0 (не тождество)."""
    need(sol, "resize_cycle_test")
    img = scene.astype(np.float64)
    err = sol["resize_cycle_test"](img, scale=2.0, method=cv2.INTER_LINEAR)
    assert err > 0, "цикл ×2 → ×1/2 не может вернуть оригинал точно"
    assert err < 5.0, "ошибка цикла должна быть малой (сейчас %.3f)" % err


def test_resize_cycle_test_identity(sol, scene):
    """При масштабе 1.0 ошибка цикла должна быть ≈ 0 (тождество)."""
    need(sol, "resize_cycle_test")
    img = scene.astype(np.float64)
    err = sol["resize_cycle_test"](img, scale=1.0, method=cv2.INTER_LINEAR)
    assert err < 1e-6, "при масштабе 1.0 ошибка должна быть ≈ 0 (сейчас %.2e)" % err


def test_resize_cycle_test_different_methods(sol, scene):
    """Разные методы интерполяции дают разную ошибку цикла."""
    need(sol, "resize_cycle_test")
    img = scene.astype(np.float64)
    err_linear = sol["resize_cycle_test"](img, scale=2.0, method=cv2.INTER_LINEAR)
    err_cubic = sol["resize_cycle_test"](img, scale=2.0, method=cv2.INTER_CUBIC)
    assert err_linear > 0 and err_cubic > 0, "оба метода должны давать ненулевую ошибку"
    # Ошибки не обязаны быть равны, но обе должны быть малыми
    assert err_linear < 10.0 and err_cubic < 10.0


def test_resize_composition_test_positive_errors(sol, scene):
    """Ошибки композиции и коммутативности должны быть > 0."""
    need(sol, "resize_composition_test")
    img = scene.astype(np.float64)
    comp_err, comm_err = sol["resize_composition_test"](img, method=cv2.INTER_LINEAR)
    assert comp_err > 0, "композиция не может быть точной для стандартных методов"
    assert comm_err > 0, "коммутативность не может быть точной для стандартных методов"


def test_resize_composition_test_small_errors(sol, scene):
    """Ошибки композиции и коммутативности должны быть малыми (< 2 кодов)."""
    need(sol, "resize_composition_test")
    img = scene.astype(np.float64)
    comp_err, comm_err = sol["resize_composition_test"](img, method=cv2.INTER_LINEAR)
    assert comp_err < 2.0, "ошибка композиции должна быть малой (сейчас %.3f)" % comp_err
    assert comm_err < 2.0, "ошибка коммутативности должна быть малой (сейчас %.3f)" % comm_err


def test_resize_composition_test_returns_tuple(sol, scene):
    """resize_composition_test возвращает кортеж из двух значений."""
    need(sol, "resize_composition_test")
    img = scene.astype(np.float64)
    result = sol["resize_composition_test"](img, method=cv2.INTER_LINEAR)
    assert isinstance(result, tuple) and len(result) == 2, \
        "должен возвращать кортеж (comp_err, comm_err)"
    comp_err, comm_err = result
    assert isinstance(comp_err, (int, float, np.floating)), "comp_err должен быть числом"
    assert isinstance(comm_err, (int, float, np.floating)), "comm_err должен быть числом"


def test_consistent_downsample_shape_and_type(sol, scene):
    """consistent_downsample возвращает массив в 2 раза меньше, в float64."""
    need(sol, "consistent_downsample")
    img = scene.astype(np.float64)
    h, w = img.shape[:2]
    result = sol["consistent_downsample"](img, scale=2.0)
    assert result.shape == (h // 2, w // 2), "размер должен быть в 2 раза меньше"
    assert result.dtype == np.float64, "результат должен быть в float64"


def test_consistent_downsample_default_sigma(sol, scene):
    """При отсутствии явного σ используется правило 0.3 * scale."""
    need(sol, "consistent_downsample")
    img = scene.astype(np.float64)
    # Должно работать без явного указания σ
    result = sol["consistent_downsample"](img, scale=2.0)
    assert result is not None and result.size > 0


def test_consistent_downsample_explicit_sigma(sol, scene):
    """При явном указании σ результат отличается от значения по умолчанию."""
    need(sol, "consistent_downsample")
    img = scene.astype(np.float64)
    result_default = sol["consistent_downsample"](img, scale=2.0)
    result_small_sigma = sol["consistent_downsample"](img, scale=2.0, sigma=0.1)
    result_large_sigma = sol["consistent_downsample"](img, scale=2.0, sigma=2.0)
    # Результаты должны различаться при разных σ
    assert not np.allclose(result_default, result_small_sigma), \
        "результаты с разным σ должны отличаться"
    assert not np.allclose(result_small_sigma, result_large_sigma), \
        "результаты с разным σ должны отличаться"