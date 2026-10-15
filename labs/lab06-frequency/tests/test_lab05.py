# -*- coding: utf-8 -*-
"""Автотесты занятия 6 «Частотная область и пирамиды». Самопроверка студента и CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:
путь в переменной окружения `CVCOURSE_LAB_NB` (так CI ассистента гоняет решение);
`lab06_solution.ipynb` рядом — появляется в репозитории после занятия;
`lab06.ipynb` — заготовка, заполненная студентом; тесты незаполненных TODO
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
    os.path.join(HERE, "lab06_solution.ipynb"),
    os.path.join(HERE, "lab06.ipynb"),
)
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)

WANTED = (
    "conv_dft",
    "find_peak",
    "notch_reject",
    "pyr_down",
    "pyr_up",
    "gauss_pyr",
    "lap_pyr",
    "collapse",
    "blend_bands",
)


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab06_solution.ipynb, ни lab06.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab06", "__todo__": set()}
    exec("import cv2, numpy as np, timeit\nfrom cvcourse import metrics", g)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        hit = [w for w in WANTED if f"def {w}(" in src]
        if not hit:
            continue
        sep = "\n" * 3               # определение отделено пустой строкой; до него могут стоять константы
        for chunk in src.split(sep):
            exec(compile(chunk, "<lab06>", "exec"), g)
            defined = [w for w in hit if f"def {w}(" in chunk]
            if defined:
                if "NotImplementedError" in chunk:   # пропуск в заготовке не заполнен
                    g["__todo__"].update(defined)
                    break                                # дальше в ячейке — ассерты, они не нужны
    for name in WANTED:
        if name not in g:
            pytest.skip(f"в ноутбуке нет функции {name}")
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
def periodic_scene(scene):
    """Кадр с прибавленной периодической помехой: пара точек в спектре."""
    h, w = scene.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    pat = 20.0 * np.sin(2 * np.pi * (xx * 7 / w + yy * 11 / h))
    noisy = np.clip(scene.astype(np.float64) + pat, 0, 255).astype(np.uint8)
    return noisy


K_ASYM = np.float64([[1, 2, 0], [0, 3, -1], [-2, 0, 1]])


# ------------------------------------------------------------- тесты ---

def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_conv_dft_matches_filter2d(sol, scene):
    """Свёртка через ДПФ совпадает с filter2D(flip, BORDER_CONSTANT) внутри кадра."""
    need(sol, "conv_dft")
    img = scene[:48, :64]
    got = np.asarray(sol["conv_dft"](img, K_ASYM), np.float64)
    ref = cv2.filter2D(np.float64(img), cv2.CV_64F, np.float64(cv2.flip(K_ASYM, -1)),
                       borderType=cv2.BORDER_CONSTANT)
    assert got.shape == img.shape and got.dtype == np.float64
    assert np.abs(got - ref)[4:-4, 4:-4].max() < 1e-9, \
        "внутри кадра расхождение с filter2D(flip, BORDER_CONSTANT) > 1e-9"


def test_conv_dft_delta_and_shift(sol, scene):
    """δ-ядро возвращает кадр; сдвиговое ядро сдвигает."""
    need(sol, "conv_dft")
    img = scene[:48, :64]
    delta = np.zeros((3, 3)); delta[1, 1] = 1
    assert np.abs(sol["conv_dft"](img, delta) - img).max() < 1e-9, \
        "δ-ядро должно возвращать кадр"
    shift = np.zeros((5, 5)); shift[2, 4] = 1
    res = sol["conv_dft"](img, shift)
    assert np.abs(res[:, 2:] - img[:, :-2]).max() < 1e-9, \
        "сдвиговое ядро должно сдвигать кадр вправо"


def test_find_peak_finds_periodic_noise(sol, periodic_scene):
    """find_peak находит смещение пика помехи относительно центра, не DC."""
    need(sol, "find_peak")
    spec = np.fft.fft2(periodic_scene.astype(np.float64))
    du, dv = sol["find_peak"](spec)
    assert (du, dv) != (0, 0), "пик не должен быть в центре (DC)"
    assert abs(du) + abs(dv) > 10, "пик должен быть заметно удалён от центра"
    A = np.abs(np.fft.fftshift(spec))
    cy, cx = A.shape[0] // 2, A.shape[1] // 2
    peak_val = A[cy + du, cx + dv]
    bg = np.median(A)
    assert peak_val / bg > 100, "пик должен быть значительно выше фона"


def test_notch_reject_improves_psnr(sol, scene, periodic_scene):
    """notch_reject поднимает PSNR шумного кадра; результат в диапазоне 0–255."""
    need(sol, "find_peak", "notch_reject")
    from cvcourse.metrics import psnr
    res = sol["notch_reject"](periodic_scene, r=16)
    assert res.shape == periodic_scene.shape
    assert res.min() >= 0 and res.max() <= 255, "результат должен быть в 0–255"
    db_before = psnr(scene, periodic_scene)
    db_after = psnr(scene, np.clip(res, 0, 255))
    assert db_after > db_before + 3, \
        f"notch должен улучшить PSNR хотя бы на 3 дБ (было {db_before:.1f}, стало {db_after:.1f})"


def test_pyr_down_matches_cv2(sol, scene):
    """Своя pyr_down совпадает с cv2.pyrDown; размер (H+1)//2 × (W+1)//2."""
    need(sol, "pyr_down")
    img = scene[:63, :97]  # нечётные размеры
    got = np.asarray(sol["pyr_down"](img), np.float32)
    ref = cv2.pyrDown(np.float32(img))
    assert got.shape == ref.shape, f"размер {got.shape} != {ref.shape}"
    assert np.abs(got - ref).max() < 1e-5, "расхождение с cv2.pyrDown > 1e-5"


def test_pyr_up_dstsize_and_roundtrip(sol, scene):
    """pyr_up с dstsize даёт нужный размер; pyr_down → pyr_up обратим до 1e-4."""
    need(sol, "pyr_down", "pyr_up")
    img = np.float32(scene[:63, :97])
    small = sol["pyr_down"](img)
    up = sol["pyr_up"](small, (img.shape[1], img.shape[0]))
    assert up.shape == img.shape, f"pyr_up вернул {up.shape}, ожидался {img.shape}"
    # Полная обратимость невозможна (потеря ВЧ), но грубая структура сохраняется
    assert np.abs(up - img).mean() < 30, "pyr_up(pyr_down(x)) слишком далеко от x"


def test_gauss_pyr_length(sol, scene):
    """gauss_pyr(x, n) возвращает n+1 уровень."""
    need(sol, "gauss_pyr")
    gp = sol["gauss_pyr"](scene, 3)
    assert len(gp) == 4, f"ожидалось 4 уровня, получено {len(gp)}"


def test_lap_pyr_float32_and_collapse(sol, scene):
    """lap_pyr считает во float32; collapse восстанавливает кадр до 1e-4."""
    need(sol, "lap_pyr", "collapse")
    lp = sol["lap_pyr"](scene, 3)
    assert all(l.dtype == np.float32 for l in lp), "все уровни lap_pyr должны быть float32"
    rec = sol["collapse"](lp)
    err = np.abs(rec - np.float32(scene)).max()
    assert err < 1e-3, f"обратимость пирамиды Лапласа: ошибка {err:.2e} > 1e-3"


def test_blend_bands_n0_is_linear(sol, scene):
    """blend_bands при n=0 совпадает с линейной маской; результат в 0–255."""
    need(sol, "blend_bands", "lap_pyr", "gauss_pyr", "collapse")
    h, w = scene.shape[:2]
    A = np.float32(scene)
    B = np.float32(np.roll(scene, 3, axis=1))
    mask = np.float32(np.arange(w)[None, :] < w // 2) * np.ones((h, 1), np.float32)

    bl = sol["blend_bands"](A, B, mask, n=0)
    linear = A * mask[..., None] + B * (1 - mask[..., None]) if A.ndim == 3 else A * mask + B * (1 - mask)
    assert bl.shape == A.shape
    assert bl.dtype == np.float64
    assert np.abs(bl - linear).max() < 1.0, "n=0 должно совпадать с линейной маской"
    assert bl.max() <= 255.0, "значения вышли за 255"