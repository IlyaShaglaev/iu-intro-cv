# -*- coding: utf-8 -*-
"""Автотесты занятия 8 «Точки интереса: детекторы». Самопроверка студента и CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:
путь в переменной окружения `CVCOURSE_LAB_NB` (так CI ассистента гоняет решение);
`lab08_solution.ipynb` рядом — появляется в репозитории после занятия;
`lab08.ipynb` — заготовка, заполненная студентом; тесты незаполненных TODO
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
    os.path.join(HERE, "lab08_solution.ipynb"),
    os.path.join(HERE, "lab08.ipynb"),
)
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)

WANTED = (
    "moment_matrix",
    "harris_response",
    "shi_tomasi_response",
    "repeatability",
)


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab08_solution.ipynb, ни lab08.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab08", "__todo__": set()}
    exec("import cv2, numpy as np, timeit\nfrom cvcourse import metrics", g)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        hit = [w for w in WANTED if "def %s(" % w in src]
        if not hit:
            continue
        sep = "\n" * 3               # определение отделено пустой строкой; до него могут стоять константы
        for chunk in src.split(sep):
            exec(compile(chunk, "<lab08>", "exec"), g)
            defined = [w for w in hit if "def %s(" % w in chunk]
            if defined:
                if "NotImplementedError" in chunk:   # пропуск в заготовке не заполнен
                    g["__todo__"].update(defined)
                    break                                # дальше в ячейке — ассерты, они не нужны
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
    """«Фотография» 120 × 160: плавный фон, три круга с резким краем, полосы."""
    yy, xx = np.mgrid[:120, :160].astype(np.float32)
    img = 110 + 50 * np.sin(xx / 19.0) * np.cos(yy / 13.0) + 0.2 * xx
    for cx, cy, r in ((40, 35, 22), (115, 80, 30), (70, 95, 14)):
        img[np.hypot(xx - cx, yy - cy) < r] += 60
    img[20:100:16, 100:150] -= 70
    return np.uint8(np.clip(img, 0, 255))


@pytest.fixture(scope="module")
def corner_scene():
    """Синтетическое изображение 128 × 128 с углом, краем и ровной областью."""
    size = 128
    img = np.zeros((size, size), np.float64)
    img[:size // 2, :size // 2] = 200       # верхний левый квадрант — угол в (64, 64)
    img[size // 2:, size // 4:3 * size // 4] = 150  # горизонтальная полоса — край
    return img


# ------------------------------------------------------------- тесты ---

def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_moment_matrix_shape_and_nonneg(sol, corner_scene):
    """moment_matrix возвращает три массива той же формы; A ≥ 0, B ≥ 0."""
    need(sol, "moment_matrix")
    A, B, C = sol["moment_matrix"](corner_scene, ksize=3, block_size=7)
    assert A.shape == corner_scene.shape
    assert B.shape == corner_scene.shape
    assert C.shape == corner_scene.shape
    assert A.min() >= 0, "A = Σ Ix² неотрицательна"
    assert B.min() >= 0, "B = Σ Iy² неотрицательна"


def test_moment_matrix_flat_region(sol):
    """На ровном кадре все компоненты M близки к нулю."""
    need(sol, "moment_matrix")
    flat = np.full((60, 80), 100, np.float64)
    A, B, C = sol["moment_matrix"](flat, ksize=3, block_size=7)
    assert A.max() < 1e-6, "на ровном кадре A должна быть ≈ 0"
    assert B.max() < 1e-6, "на ровном кадре B должна быть ≈ 0"
    assert np.abs(C).max() < 1e-6, "на ровном кадре C должна быть ≈ 0"


def test_moment_matrix_corner_vs_edge(sol, corner_scene):
    """На углу A+B больше, чем на крае; на крае больше, чем на ровном."""
    need(sol, "moment_matrix")
    A, B, C = sol["moment_matrix"](corner_scene, ksize=3, block_size=7)
    corner_val = A[64, 64] + B[64, 64]
    edge_val = A[96, 64] + B[96, 64]
    flat_val = A[110, 20] + B[110, 20]
    assert corner_val > edge_val * 0.5, "на углу A+B должно быть больше, чем на крае"
    assert edge_val > flat_val * 5, "на крае A+B должно быть больше, чем на ровном"


def test_harris_response_shape_and_corner(sol, corner_scene):
    """harris_response: R > 0 на углу, форма совпадает, тип float64."""
    need(sol, "harris_response", "moment_matrix")
    R = sol["harris_response"](corner_scene, ksize=3, block_size=7, k=0.04)
    assert R.shape == corner_scene.shape
    assert R.dtype == np.float64
    assert R[64, 64] > 0, "на углу R должно быть > 0"


def test_harris_response_corner_vs_flat(sol, corner_scene):
    """На углу R значительно больше, чем на ровном участке."""
    need(sol, "harris_response", "moment_matrix")
    R = sol["harris_response"](corner_scene, ksize=3, block_size=7, k=0.04)
    assert R[64, 64] > R[110, 20], "R на углу должно быть больше, чем на ровном"


def test_harris_correlates_with_cv2(sol, scene):
    """harris_response коррелирует с cv2.cornerHarris > 0.8 на реальном кадре."""
    need(sol, "harris_response", "moment_matrix")
    img = scene.astype(np.float64)
    R_mine = sol["harris_response"](img, ksize=3, block_size=3, k=0.04)
    R_cv = cv2.cornerHarris(np.float32(img), blockSize=3, ksize=3, k=0.04)
    corr = np.corrcoef(R_mine.ravel(), R_cv.ravel())[0, 1]
    assert corr > 0.8, "корреляция с cv2.cornerHarris должна быть > 0.8 (сейчас %.3f)" % corr


def test_shi_tomasi_response_corner_gt_edge_gt_flat(sol, corner_scene):
    """Ши-Томаси: R(угол) > R(край) > R(ровное)."""
    need(sol, "shi_tomasi_response", "moment_matrix")
    R = sol["shi_tomasi_response"](corner_scene, ksize=3, block_size=7)
    assert R.shape == corner_scene.shape
    R_corner = R[64, 64]
    R_edge = R[96, 64]
    R_flat = R[110, 20]
    assert R_corner > R_edge, "на углу min(λ₁,λ₂) больше, чем на крае"
    assert R_edge > R_flat, "на крае min(λ₁,λ₂) больше, чем на ровном"


def test_repeatability_zero_angle(sol, scene):
    """При повороте на 0° повторяемость должна быть ≈ 1.0."""
    need(sol, "repeatability", "harris_response", "moment_matrix")
    img = scene.astype(np.float64)
    rep = sol["repeatability"](img, angle=0, block_size=7, ksize=3, k=0.04, threshold_frac=0.01)
    assert rep > 0.9, "при повороте на 0° повторяемость должна быть > 0.9 (сейчас %.2f)" % rep


def test_repeatability_rotation_invariance(sol, scene):
    """При повороте на 30° повторяемость Харриса должна быть заметной (> 0.2)."""
    need(sol, "repeatability", "harris_response", "moment_matrix")
    img = scene.astype(np.float64)
    rep = sol["repeatability"](img, angle=30, block_size=7, ksize=3, k=0.04, threshold_frac=0.01)
    assert rep > 0.2, "при повороте на 30° повторяемость Харриса должна быть > 0.2 (сейчас %.2f)" % rep