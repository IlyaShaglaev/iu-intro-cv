# -*- coding: utf-8 -*-
"""ДЗ: шаблон. Заполните TODO; команды и контракт — в README.md."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Не удалось прочитать ' + str(path))
    return image


def board_points(columns, rows, square_m):
    """Внутренние углы шахматки: row-major, все расстояния в метрах."""
    # сетка XY, масштаб, Z=0
    raise NotImplementedError("TODO 1")


def reprojection_errors(objects, images, K, distortion, rvecs, tvecs):
    """Для каждой позы sqrt(mean(||проекция-наблюдение||²)), в пикселях."""
    # projectPoints и RMS по двумерным евклидовым ошибкам
    raise NotImplementedError("TODO 2")


def calibrate(objects, images, image_size):
    """Вернуть RMS, K, dist, rvecs, tvecs, per_image; image_size=(W,H)."""
    # проверить набор, calibrateCamera, ошибки по позам
    raise NotImplementedError("TODO 3")


def plane_measure(pixels, K, distortion, rvec, tvec):
    """Луч из пикселя пересечь с Z=0 доски; вернуть точки в СК доски (метры)."""
    # undistortPoints, R.T, центр камеры, пересечение лучей с плоскостью
    raise NotImplementedError("TODO 4")


def disparity_metrics(prediction, truth, valid):
    """bad-2.0 в %, RMSE в px; маска GT обязательна, невалидный prediction — ошибка."""
    # маска GT, покрытие, bad2 с учётом пропусков, RMSE
    raise NotImplementedError("TODO 5")


def stereo_depth(disparity, focal_px, baseline_m, doffs=0.0):
    """Z=fB/(d+doffs); неизвестная/неположительная диспаратность → NaN."""
    # валидация единиц, маска, глубина без деления на ноль
    raise NotImplementedError("TODO 6")


def sgbm(left, right, num_disparities=32, block_size=5):
    """Вход — ректифицированная пара. OpenCV fixed point /16 → пиксели."""
    # параметры SGBM, compute, float /16, маска пропусков
    raise NotImplementedError("TODO 7")


def point_cloud(depth, K):
    """Плотное XYZ в системе левой камеры; пиксели отсчитываются от (0,0)."""
    # сетка uv, Z, X=(u-cx)Z/fx, Y=(v-cy)Z/fy
    raise NotImplementedError("TODO 8")


def write_ply(path, xyz, colors):
    valid = np.isfinite(xyz).all(axis=-1)
    points, rgb = xyz[valid], colors[valid][:, ::-1]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(points))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(points, rgb):
            f.write('%.6f %.6f %.6f %d %d %d\n' % (*p, *c))


def detect_board(folder, columns, rows, square_m):
    objects, images, names, size = [], [], [], None
    for p in sorted(Path(folder).glob('*')):
        if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        gray = cv2.cvtColor(read_image(p), cv2.COLOR_BGR2GRAY)
        current = (gray.shape[1], gray.shape[0])
        if size is not None and current != size:
            raise ValueError('Все фото калибровки должны иметь один размер')
        size = current
        ok, corners = cv2.findChessboardCornersSB(gray, (columns, rows))
        if ok:
            objects.append(board_points(columns, rows, square_m))
            images.append(corners)
            names.append(p.name)
    return objects, images, size, names


def calibration_report(objects, images, size, max_error=None):
    before = calibrate(objects, images, size)
    keep = np.ones(len(objects), bool) if max_error is None else before[-1] <= max_error
    if keep.sum() < 3:
        raise ValueError('После отбраковки осталось меньше трёх поз')
    selected = calibrate([o for o,k in zip(objects,keep) if k],
                         [p for p,k in zip(images,keep) if k], size)
    rms, K, distortion, rotations, translations, errors = selected
    return dict(rms_before=float(before[0]), rms_after=float(rms), K=K.tolist(),
                distortion=distortion.ravel().tolist(), per_image_before_px=before[-1].tolist(),
                per_image_after_px=errors.tolist(), kept_indices=np.flatnonzero(keep).tolist(),
                rvecs=[r.ravel().tolist() for r in rotations],
                tvecs_m=[t.ravel().tolist() for t in translations])


def main():
    parser = argparse.ArgumentParser(description='HW4: калибровка и 3D')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('calibrate')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--observations', help='NPZ: object_points, image_points, image_size')
    source.add_argument('--images', help='Папка фото шахматки')
    p.add_argument('--columns', type=int, default=7)
    p.add_argument('--rows', type=int, default=5)
    p.add_argument('--square-m', type=float, default=0.03)
    p.add_argument('--max-error', type=float)
    p.add_argument('-o', '--output', required=True)
    p = sub.add_parser('stereo')
    p.add_argument('sample', help='Папка left.png/right.png/gt.npz')
    p.add_argument('--num-disparities', type=int, default=32)
    p.add_argument('--block-size', type=int, default=5)
    p.add_argument('-o', '--output', required=True, help='Каталог результата')
    p = sub.add_parser('measure')
    p.add_argument('spec', help='JSON: K, distortion, rvec, tvec_m, pixels, truth_length_m')
    p = sub.add_parser('pose')
    p.add_argument('image')
    p.add_argument('--calibration', required=True, help='JSON из команды calibrate')
    p.add_argument('--columns', type=int, default=7)
    p.add_argument('--rows', type=int, default=5)
    p.add_argument('--square-m', type=float, default=0.03)
    p.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    if args.command == 'calibrate':
        if args.observations:
            with np.load(args.observations, allow_pickle=False) as data:
                objects, images, size = data['object_points'], data['image_points'], data['image_size']
        else:
            objects, images, size, _ = detect_board(args.images, args.columns, args.rows, args.square_m)
        result = calibration_report(objects, images, size, args.max_error)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    elif args.command == 'stereo':
        folder = Path(args.sample)
        left, right = read_image(folder/'left.png'), read_image(folder/'right.png')
        with np.load(folder/'gt.npz', allow_pickle=False) as data:
            truth, valid, K = data['disparity'], data['valid'], data['K']
            baseline, doffs = float(data['baseline_m']), float(data['doffs'])
        disparity = sgbm(left, right, args.num_disparities, args.block_size)
        result = disparity_metrics(disparity, truth, valid)
        depth = stereo_depth(disparity, K[0,0], baseline, doffs)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out/'result.npz', disparity=disparity, depth_m=depth)
        write_ply(out/'cloud.ply', point_cloud(depth, K), left)
        (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    elif args.command == 'pose':
        data = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
        K, distortion = np.float64(data['K']), np.float64(data['distortion'])
        gray = cv2.cvtColor(read_image(args.image), cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, (args.columns, args.rows))
        if not found:
            raise ValueError('Доска не найдена на кадре измерения')
        objects = board_points(args.columns, args.rows, args.square_m)
        ok, rvec, tvec = cv2.solvePnP(objects, corners, K, distortion)
        if not ok:
            raise ValueError('solvePnP не нашёл позу')
        result = dict(K=K.tolist(), distortion=distortion.tolist(), rvec=rvec.ravel().tolist(),
                      tvec_m=tvec.ravel().tolist(), board_origin_distance_m=float(np.linalg.norm(tvec)),
                      reprojection_rms_px=float(reprojection_errors([objects], [corners], K,
                                               distortion, [rvec], [tvec])[0]))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    else:
        data = json.loads(Path(args.spec).read_text(encoding='utf-8'))
        xyz = plane_measure(data['pixels'], np.float64(data['K']), np.float64(data['distortion']),
                            np.float64(data['rvec']), np.float64(data['tvec_m']))
        length = float(np.linalg.norm(xyz[1] - xyz[0]))
        result = dict(points_m=xyz.tolist(), length_m=length,
                      relative_error_pct=100*abs(length-data['truth_length_m'])/data['truth_length_m'],
                      board_origin_distance_m=float(np.linalg.norm(data['tvec_m'])))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
