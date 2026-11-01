# -*- coding: utf-8 -*-
"""Автотесты занятия 9 «Дескрипторы и сопоставление». Самопроверка студента и CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:
путь в переменной окружения `CVCOURSE_LAB_NB` (так CI ассистента гоняет решение);
`lab09_solution.ipynb` рядом — появляется в репозитории после занятия;
`lab09.ipynb` — заготовка, заполненная студентом; тесты незаполненных TODO
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
    os.path.join(HERE, "lab09_solution.ipynb"),
    os.path.join(HERE, "lab09.ipynb"),
)
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)

WANTED = (
    "bf_match_with_ratio",
    "orb_flann_match",
    "verify_matches",
    "match_pipeline",
)


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab09_solution.ipynb, ни lab09.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab09", "__todo__": set()}
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
            exec(compile(chunk, "<lab09>", "exec"), g)
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
def scene_pair(scene):
    """Два кадра: исходный и повёрнутый на 30° с масштабом ×1.3."""
    h, w = scene.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.3)
    rotated = cv2.warpAffine(scene, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return scene.astype(np.float32), rotated.astype(np.float32)


@pytest.fixture(scope="module")
def sift_descriptors(scene_pair):
    """SIFT-дескрипторы для пары кадров."""
    sift = cv2.SIFT_create(nfeatures=200)
    kp1, desc1 = sift.detectAndCompute(scene_pair[0], None)
    kp2, desc2 = sift.detectAndCompute(scene_pair[1], None)
    return kp1, kp2, desc1, desc2


# ------------------------------------------------------------- тесты ---

def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_bf_match_returns_dmatch_list(sol, sift_descriptors):
    """bf_match_with_ratio возвращает список cv2.DMatch."""
    need(sol, "bf_match_with_ratio")
    _, _, desc1, desc2 = sift_descriptors
    matches = sol["bf_match_with_ratio"](desc1, desc2, ratio=0.75)
    assert isinstance(matches, list), "результат должен быть списком"
    assert all(isinstance(m, cv2.DMatch) for m in matches), \
        "элементы списка должны быть cv2.DMatch"


def test_bf_match_ratio_filters(sol, sift_descriptors):
    """ratio=0.75 даёт меньше совпадений, чем ratio=1.0; результат не пуст."""
    need(sol, "bf_match_with_ratio")
    _, _, desc1, desc2 = sift_descriptors
    strict = sol["bf_match_with_ratio"](desc1, desc2, ratio=0.75)
    loose = sol["bf_match_with_ratio"](desc1, desc2, ratio=1.0)
    assert len(strict) > 0, "при ratio=0.75 должно быть хотя бы одно совпадение"
    assert len(strict) < len(loose), \
        "ratio=0.75 должно давать меньше совпадений, чем ratio=1.0 (%d vs %d)" % (
            len(strict), len(loose))


def test_orb_flann_match_returns_tuple(sol, scene_pair):
    """orb_flann_match возвращает (kp1, kp2, good_matches)."""
    need(sol, "orb_flann_match")
    img1, img2 = scene_pair
    result = sol["orb_flann_match"](img1, img2, ratio=0.75)
    assert isinstance(result, tuple) and len(result) == 3, \
        "должен возвращать кортеж (kp1, kp2, good_matches)"
    kp1, kp2, good = result
    assert len(kp1) > 0 and len(kp2) > 0, "ORB должен найти ключевые точки"
    assert isinstance(good, list), "good_matches должен быть списком"


def test_orb_flann_match_has_matches(sol, scene_pair):
    """FLANN + ratio test на ORB даёт хотя бы одно совпадение."""
    need(sol, "orb_flann_match")
    img1, img2 = scene_pair
    _, _, good = sol["orb_flann_match"](img1, img2, ratio=0.75)
    assert len(good) > 0, "FLANN + ratio test должен дать хотя бы одно совпадение"


def test_verify_matches_returns_homography(sol, sift_descriptors):
    """verify_matches возвращает гомографию 3×3 и маску инлайеров."""
    need(sol, "verify_matches", "bf_match_with_ratio")
    kp1, kp2, desc1, desc2 = sift_descriptors
    matches = sol["bf_match_with_ratio"](desc1, desc2, ratio=0.75)
    if len(matches) < 4:
        pytest.skip("слишком мало совпадений после ratio test для теста RANSAC")
    H, mask, inliers = sol["verify_matches"](kp1, kp2, matches, ransac_thresh=3.0)
    assert H is not None and H.shape == (3, 3), "гомография должна быть матрицей 3×3"
    assert isinstance(inliers, list), "inlier_matches должен быть списком"


def test_verify_matches_inliers_subset(sol, sift_descriptors):
    """RANSAC не добавляет совпадения: инлайеры ⊆ исходные matches."""
    need(sol, "verify_matches", "bf_match_with_ratio")
    kp1, kp2, desc1, desc2 = sift_descriptors
    matches = sol["bf_match_with_ratio"](desc1, desc2, ratio=0.75)
    if len(matches) < 4:
        pytest.skip("слишком мало совпадений после ratio test для теста RANSAC")
    _, _, inliers = sol["verify_matches"](kp1, kp2, matches, ransac_thresh=3.0)
    assert len(inliers) <= len(matches), "RANSAC не может добавить совпадения"
    assert len(inliers) > 0, "после RANSAC должны остаться инлайеры"


def test_match_pipeline_sift(sol, scene_pair):
    """match_pipeline с SIFT возвращает гомографию и ≥4 инлайнера."""
    need(sol, "match_pipeline")
    img1, img2 = scene_pair
    kp1, kp2, inliers, H = sol["match_pipeline"](img1, img2, detector='sift')
    assert H is not None and H.shape == (3, 3), "конвейер должен вернуть гомографию"
    assert len(inliers) >= 4, "для гомографии нужно минимум 4 инлайнера (сейчас %d)" % len(inliers)


def test_match_pipeline_orb(sol, scene_pair):
    """match_pipeline с ORB возвращает результат без ошибок."""
    need(sol, "match_pipeline")
    img1, img2 = scene_pair
    kp1, kp2, inliers, H = sol["match_pipeline"](img1, img2, detector='orb')
    assert isinstance(inliers, list), "inlier_matches должен быть списком"
    # ORB может не найти достаточно совпадений на маленькой синтетике —
    # проверяем только отсутствие ошибок и корректность типов
    if H is not None:
        assert H.shape == (3, 3), "гомография должна быть матрицей 3×3"