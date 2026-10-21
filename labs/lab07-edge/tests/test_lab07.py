# -*- coding: utf-8 -*-
"""Автотесты занятия 7 «Границы, линии и робастное оценивание». Самопроверка студента и CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:
путь в переменной окружения `CVCOURSE_LAB_NB` (так CI ассистента гоняет решение);
`lab07_solution.ipynb` рядом — появляется в репозитории после занятия;
`lab07.ipynb` — заготовка, заполненная студентом; тесты незаполненных TODO
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
    os.path.join(HERE, "lab07_solution.ipynb"),
    os.path.join(HERE, "lab07.ipynb"),
)
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)

WANTED = (
    "gradient_mag",
    "canny_manual",
    "hough_accum",
    "ransac_line",
)


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab07_solution.ipynb, ни lab07.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab07", "__todo__": set()}
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
            exec(compile(chunk, "<lab07>", "exec"), g)
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
def noisy_scene(scene):
    """Шумный кадр σ = 10 — модель демо 1.1 лекции."""
    return np.clip(scene.astype(np.float64)
                   + np.random.default_rng(0).normal(0, 10, scene.shape),
                   0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def line_points():
    """Точки прямой θ = 35°, ρ = 120 с шумом σ = 2 и 30 % выбросов (демо 2.1 и 3.1)."""
    rng = np.random.default_rng(1)
    th0, r0 = np.radians(35), 120
    s = np.linspace(-180, 180, 300)
    t = rng.normal(0, 2.0, 300)
    xs = r0 * np.cos(th0) - s * np.sin(th0) + t
    ys = r0 * np.sin(th0) + s * np.cos(th0) + t
    xs = np.r_[xs, rng.uniform(0, 400, 200)]
    ys = np.r_[ys, rng.uniform(0, 400, 200)]
    return np.column_stack([xs, ys])


# ------------------------------------------------------------- тесты ---

def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_gradient_mag_shape_and_positive(sol, noisy_scene):
    """Магнитуда градиента — массив того же размера, неотрицательный, в float64."""
    need(sol, "gradient_mag")
    got = np.asarray(sol["gradient_mag"](noisy_scene, ksize=3, sigma=0), np.float64)
    assert got.shape == noisy_scene.shape
    assert got.dtype == np.float64
    assert got.min() >= 0, "магнитуда градиента не может быть отрицательной"


def test_gradient_mag_smoothing_reduces_noise(sol, noisy_scene):
    """Сглаживание перед производной уменьшает дисперсию градиента на шумном кадре."""
    need(sol, "gradient_mag")
    g = np.asarray(sol["gradient_mag"](noisy_scene, ksize=3, sigma=0), np.float64)
    g_s = np.asarray(sol["gradient_mag"](noisy_scene, ksize=3, sigma=1.0), np.float64)
    assert g_s.std() < g.std(), (
        "сглаживание должно уменьшать дисперсию градиента на шумном кадре "
        "(сейчас %.1f против %.1f)" % (g_s.std(), g.std()))


def test_gradient_mag_flat_region(sol):
    """На ровном кадре градиент близок к нулю."""
    need(sol, "gradient_mag")
    flat = np.full((40, 60), 100, np.uint8)
    got = np.asarray(sol["gradient_mag"](flat, ksize=3, sigma=0), np.float64)
    assert got.shape == flat.shape
    assert got.max() < 1.0, "на ровном кадре градиент должен быть близок к нулю"


def test_canny_manual_matches_cv2(sol, noisy_scene):
    """Canny вручную совпадает с cv2.Canny (с тем же сглаживанием) более чем на 85 % пикселей."""
    need(sol, "canny_manual")
    sigma, lo, hi = 1.0, 30, 100
    got = sol["canny_manual"](noisy_scene, sigma, lo, hi)
    ref = cv2.Canny(cv2.GaussianBlur(noisy_scene, (0, 0), sigma), lo, hi)
    assert got.shape == noisy_scene.shape and got.dtype == np.uint8
    overlap = (got == ref).mean()
    assert overlap > 0.85, (
        "доля совпавших пикселей с cv2.Canny должна быть > 0.85 (сейчас %.3f)" % overlap)


def test_canny_manual_binary(sol, noisy_scene):
    """Результат Canny вручную — бинарная карта: только 0 или 255."""
    need(sol, "canny_manual")
    got = sol["canny_manual"](noisy_scene, 1.0, 30, 100)
    assert set(np.unique(got)).issubset({0, 255}), "результат должен быть бинарным: 0 или 255"
    assert (got > 0).any(), "на кадре с краями должны быть найдены края"


def test_hough_accum_shape_and_peak(sol, line_points):
    """Аккумулятор Хафа: 180 строк при θ_res = 1; пик около θ = 35°, ρ = 120."""
    need(sol, "hough_accum")
    acc, thetas, rhos = sol["hough_accum"](line_points, theta_res=1, rho_res=1)
    assert acc.shape[0] == 180, "при θ_res = 1 должно быть 180 строк в аккумуляторе"
    assert acc.dtype in (np.int32, np.int64), "аккумулятор — целочисленный"
    i, j = np.unravel_index(acc.argmax(), acc.shape)
    th_peak = np.rad2deg(thetas[i])
    rho_peak = rhos[j]
    assert acc.max() > 50, "пик должен быть заметным (больше 50 голосов)"
    assert abs(th_peak - 35) < 5, "пик должен быть около θ = 35° (сейчас %.0f°)" % th_peak
    assert abs(rho_peak - 120) < 10, "пик должен быть около ρ = 120 (сейчас %d)" % rho_peak


def test_ransac_line_finds_line(sol, line_points):
    """RANSAC находит модель прямой с долей инлайеров больше 50 %."""
    need(sol, "ransac_line")
    model, inliers = sol["ransac_line"](line_points, threshold=4.0)
    assert model is not None, "RANSAC должен найти модель"
    assert inliers.dtype == bool and len(inliers) == len(line_points)
    assert inliers.mean() > 0.5, (
        "доля инлайеров должна быть > 50 %% (сейчас %.0f %%)" % (100 * inliers.mean()))


def test_ransac_line_angle(sol, line_points):
    """Угол найденной прямой близок к истинному θ = 35°."""
    need(sol, "ransac_line")
    model, inliers = sol["ransac_line"](line_points, threshold=4.0)
    a, b, c = model
    angle = np.rad2deg(np.arctan2(-a, b)) % 180
    assert abs(angle - 35) < 10 or abs(angle - 35 + 180) < 10, (
        "угол прямой должен быть около 35° (сейчас %.0f°)" % angle)


def test_ransac_inliers_consistent_with_model(sol, line_points):
    """Все инлайеры лежат ближе порога к найденной прямой."""
    need(sol, "ransac_line")
    model, inliers = sol["ransac_line"](line_points, threshold=4.0)
    a, b, c = model
    dist = np.abs(a * line_points[:, 0] + b * line_points[:, 1] + c) / np.sqrt(a ** 2 + b ** 2)
    assert dist[inliers].max() < 4.0, "все инлайеры должны лежать ближе порога к прямой"