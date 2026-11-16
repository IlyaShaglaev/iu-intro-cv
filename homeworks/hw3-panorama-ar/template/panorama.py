# -*- coding: utf-8 -*-
"""ДЗ: шаблон. Заполните TODO; команды и контракт — в README.md."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Не удалось прочитать изображение: ' + str(path))
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError('Ошибка кодирования: ' + str(path))
    encoded.tofile(path)


def normalize_points(points):
    """Нормализация Хартли: центроид 0, среднее расстояние sqrt(2)."""
    # центроид, масштаб, матрица T, преобразованные точки
    raise NotImplementedError("TODO 1")


def dlt(src, dst):
    """Нормализованный DLT; H отображает src в dst, H[2,2]=1."""
    # A, SVD, денормализация; проверить ранг
    raise NotImplementedError("TODO 2")


def project(points, H):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    q = np.column_stack([points, np.ones(len(points))]) @ np.asarray(H).T
    valid = np.abs(q[:, 2]) > 1e-12
    out = np.full((len(points), 2), np.inf)
    out[valid] = q[valid, :2] / q[valid, 2:3]
    return out


def ransac_homography(src, dst, threshold=3.0, iterations=500, seed=0):
    """Свой цикл RANSAC, свой DLT, маска инлаеров в исходном порядке."""
    # выборка 4, ошибка репроекции, лучшая маска, уточнение
    raise NotImplementedError("TODO 3")


def match_points(image1, image2, detector='sift', ratio=0.75):
    """SIFT/ORB и ratio test; вернуть две согласованные Nx2 матрицы."""
    # детектор, подходящая норма, knnMatch и ratio test
    raise NotImplementedError("TODO 4")


def image_corners(image):
    h, w = image.shape[:2]
    return np.float64([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])


def feather_blend(images, transforms):
    """H_i: кадр i → система первого кадра; чёрный цвет не означает пустоту."""
    # общий холст, сдвиг, маски покрытия и взвешенная сумма
    raise NotImplementedError("TODO 5")


def smooth_homography(previous, current, corners, alpha=0.4):
    """EMA проекций углов: alpha=1 означает текущий кадр без сглаживания."""
    # сгладить углы, затем восстановить H своим DLT
    raise NotImplementedError("TODO 6")


def overlay(frame, texture, H, opacity=0.8):
    """H переводит углы texture в координаты кадра."""
    # варп текстуры и отдельной маски, альфа-композитинг
    raise NotImplementedError("TODO 7")


def stitch(images, detector='sift', ratio=0.75, seed=0):
    transforms, reports = [np.eye(3)], []
    for i in range(1, len(images)):
        a, b = match_points(images[i], images[i-1], detector, ratio)
        H, mask = ransac_homography(a, b, seed=seed)
        transforms.append(transforms[-1] @ H)
        error = np.linalg.norm(project(a[mask], H) - b[mask], axis=1).mean()
        reports.append(dict(pair=[i, i-1], matches=len(a), inlier_ratio=float(mask.mean()),
                            reprojection_px=float(error)))
    panorama, _ = feather_blend(images, transforms)
    return panorama, reports


def ar_video(reference, texture, source, output=None, detector='sift', alpha=0.4, display=False):
    source = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError('Не удалось открыть видео/камеру')
    writer, previous, count, tracked = None, None, 0, 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            try:
                a, b = match_points(reference, frame, detector)
                H, _ = ransac_homography(a, b, iterations=150)
                previous = smooth_homography(previous, H, image_corners(reference), alpha)
                # Текстура масштабируется в размер опорного изображения.
                tex = cv2.resize(texture, (reference.shape[1], reference.shape[0]))
                frame = overlay(frame, tex, previous)
                tracked += 1
            except ValueError:
                previous = None  # При потере цели старую H не рисуем.
            if output:
                if writer is None:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 15
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise ValueError('Не удалось открыть выходное видео')
                writer.write(frame)
            if display:
                cv2.imshow('AR: Esc = выход', frame)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    if not count:
        raise ValueError('Видео не содержит кадров')
    return dict(frames=count, tracked=tracked, fps=count / elapsed)


def main():
    parser = argparse.ArgumentParser(description='HW3: панорама и AR')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('panorama')
    p.add_argument('images', nargs='+')
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--detector', choices=['sift', 'orb'], default='sift')
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('dlt-check')
    p.add_argument('correspondences', help='JSON: src, dst (точные пары)')
    p = sub.add_parser('ar')
    p.add_argument('reference')
    p.add_argument('texture')
    p.add_argument('--source', required=True)
    p.add_argument('-o', '--output')
    p.add_argument('--alpha', type=float, default=0.4)
    p.add_argument('--display', action='store_true')
    args = parser.parse_args()
    if args.command == 'panorama':
        image, report = stitch([read_image(p) for p in args.images], args.detector, seed=args.seed)
        write_image(args.output, image)
        result = dict(shape=list(image.shape), pairs=report)
    elif args.command == 'dlt-check':
        data = json.loads(Path(args.correspondences).read_text(encoding='utf-8'))
        src, dst = np.float64(data['src']), np.float64(data['dst'])
        mine = dlt(src, dst)
        library, _ = cv2.findHomography(src, dst, method=0)
        delta = float(np.linalg.norm(mine / mine[2, 2] - library / library[2, 2]))
        result = dict(frobenius=delta, passed=delta < 1e-6)
    else:
        result = ar_video(read_image(args.reference), read_image(args.texture), args.source,
                          args.output, alpha=args.alpha, display=args.display)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
