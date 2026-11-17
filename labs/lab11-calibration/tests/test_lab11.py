"""Offline tests; set CVCOURSE_LAB_NB to test the private reference notebook.

Unfilled student functions skip their numerical checks. Missing definitions,
imports and ordinary runtime errors fail. No code from a private clone is used
by default: all synthetic data comes from the selected notebook.
"""
import ast
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
DEFAULT = HERE / ('lab11_solution.ipynb' if (HERE / 'lab11_solution.ipynb').is_file() else 'lab11.ipynb')
NB = Path(os.environ.get('CVCOURSE_LAB_NB', DEFAULT))


@pytest.fixture(scope='module')
def sol():
    notebook = json.loads(NB.read_text(encoding='utf-8'))
    namespace = {'__name__': 'lab11_tests', '__todo__': set()}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code' or 'lab11-definitions' not in cell.get('metadata', {}).get('tags', []):
            continue
        source = ''.join(cell['source'])
        exec(compile(source, str(NB), 'exec'), namespace)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                and isinstance(n.exc.func, ast.Name) and n.exc.func.id == 'NotImplementedError'
                for n in ast.walk(node)
            ):
                namespace['__todo__'].add(node.name)
    for name in ('make_scene', 'collect_views', 'calibrate_trim', 'pose_cube', 'draw_cube', 'CUBE'):
        assert name in namespace, f'Missing definition: {name}'
    return namespace


def need(sol, *names):
    todo = sol['__todo__'].intersection(names)
    if todo:
        pytest.skip('Unfilled TODO: ' + ', '.join(sorted(todo)))


@pytest.fixture(scope='module')
def scene(sol):
    return sol['make_scene']()


@pytest.fixture(scope='module')
def observed(sol, scene):
    need(sol, 'collect_views')
    board, frames, _, _, size, _ = scene
    blank = np.full((size[1], size[0]), 190, np.uint8)
    return sol['collect_views'](frames + [blank], board)


@pytest.fixture(scope='module')
def calibrated(sol, scene, observed):
    need(sol, 'calibrate_trim')
    objects, images, accepted, _ = observed
    corrupted = [x.copy() for x in images]
    index = accepted.index(7)
    corrupted[index] += np.random.default_rng(11).normal(0, 4, corrupted[index].shape).astype(np.float32)
    result = sol['calibrate_trim'](objects, corrupted, scene[4])
    return result, objects, corrupted


def test_definitions_and_environment(sol):
    assert cv2.__version__ == '4.14.0'
    assert callable(sol['make_scene'])


def test_rendered_data_geometry_and_repeatability(sol, scene):
    board, frames, K, dist, size, poses = scene
    other = sol['make_scene']()
    assert len(frames) == len(poses) == 20 and size == (640, 480)
    assert all(x.dtype == np.uint8 and x.shape == (480, 640) for x in frames)
    assert all(np.array_equal(a, b) for a, b in zip(frames, other[1]))
    assert np.linalg.norm(dist) > 0
    assert K.shape == (3, 3) and K[0, 0] != K[1, 1]
    # Check the renderer against independent forward geometry, not its own maps.
    for i in (0, 9, 19):
        corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(frames[i])
        projected, _ = cv2.projectPoints(board.getChessboardCorners()[ids.ravel()], *poses[i], K, dist)
        rms = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners.reshape(-1, 2))**2, axis=1)))
        assert rms < .75


def test_detect_views_and_reject_blank(observed):
    objects, images, accepted, rejected = observed
    assert accepted == list(range(20)) and rejected == [20]
    assert len(objects) == len(images) == 20
    assert all(a.dtype == b.dtype == np.float32 and len(a) == len(b) == 24 for a, b in zip(objects, images))
    assert all(np.allclose(x.reshape(-1, 3)[:, 2], 0) for x in objects)


def test_detect_partial_board_uses_ids(sol, scene):
    need(sol, 'collect_views')
    board, frames, _, _, _, _ = scene
    partial = frames[7].copy()
    partial[:, 315:] = 190
    obj, img, accepted, rejected = sol['collect_views']([partial], board)
    assert accepted == [0] and rejected == [] and 6 <= len(obj[0]) < 24
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(partial)
    np.testing.assert_allclose(obj[0].reshape(-1, 3), board.getChessboardCorners()[ids.ravel()])
    np.testing.assert_allclose(img[0], corners)


def test_detect_bad_input(sol, scene):
    need(sol, 'collect_views')
    board, frames, _, _, _, _ = scene
    for bad in ([], [frames[0], frames[1][:200]], [frames[0].astype(np.float32)], [None]):
        with pytest.raises(ValueError):
            sol['collect_views'](bad, board)


def test_trim_outlier_and_recover_intrinsics(calibrated, scene):
    result, _, _ = calibrated
    keep = result['keep']
    assert keep.dtype == bool and keep.shape == (20,)
    assert np.flatnonzero(~keep).tolist() == [7]
    assert .8 < result['before'][0] < 1.4
    assert .15 < result['after'][0] < .5
    K = result['after'][1]
    assert np.max(np.abs(np.diag(K)[:2] / np.diag(scene[2])[:2] - 1)) < .02
    assert np.max(np.abs(K[:2, 2] - scene[2][:2, 2])) < 12


def test_errors_are_point_rms_and_refit_uses_retained_views(calibrated):
    result, objects, images = calibrated
    for name, mask in (('before', np.ones(len(objects), bool)), ('after', result['keep'])):
        rms, K, dist, rvecs, tvecs = result[name]
        independent = []
        for i, rv, tv in zip(np.flatnonzero(mask), rvecs, tvecs):
            projected, _ = cv2.projectPoints(objects[i], rv, tv, K, dist)
            residual = projected.reshape(-1, 2) - images[i].reshape(-1, 2)
            independent.append(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        assert len(rvecs) == len(tvecs) == mask.sum()
        np.testing.assert_allclose(result['errors_' + name], independent, atol=1e-5)
        assert abs(rms - np.sqrt(np.mean(np.square(independent)))) < 1e-5


def test_clean_observations_keep_all_views(sol, observed, scene):
    need(sol, 'calibrate_trim')
    result = sol['calibrate_trim'](observed[0], observed[1], scene[4])
    assert result['keep'].all()
    assert result['after'][0] < .5


def test_calibration_rejects_insufficient_views(sol, scene):
    need(sol, 'calibrate_trim')
    with pytest.raises(ValueError):
        sol['calibrate_trim']([], [], scene[4])


def test_pose_uses_metric_board_and_correct_coordinate_system(sol, scene, observed, calibrated):
    need(sol, 'pose_cube')
    result, _, _ = calibrated
    K, dist = result['after'][1:3]
    pose = sol['pose_cube'](scene[1][0], observed[0][0], observed[1][0], K, dist)
    rv, tv = pose['rvec'], pose['tvec']
    expected, _ = cv2.projectPoints(sol['CUBE'], rv, tv, K, np.zeros(5))
    np.testing.assert_allclose(pose['cube_pixels'], expected.reshape(8, 2), atol=.005)
    assert 0 < pose['rms'] < .6
    assert np.linalg.norm(tv.ravel() - scene[5][0][1]) < .02
    assert ((sol['CUBE'] @ cv2.Rodrigues(rv)[0].T + tv.ravel())[:, 2] > 0).all()
    assert pose['overlay'].shape == (480, 640, 3) and pose['overlay'].dtype == np.uint8
    corrected = cv2.undistort(scene[1][0], K, dist, None, K)
    assert np.count_nonzero(pose['overlay'][:, :, 2] != corrected) > 50


def test_pose_on_independent_heldout_geometry(sol, calibrated, scene):
    need(sol, 'pose_cube')
    K, dist = calibrated[0]['after'][1:3]
    points = scene[0].getChessboardCorners().reshape(-1, 1, 3)
    rv = np.array([.22, -.18, .08])
    tv = np.array([-.09, -.05, .55])
    pixels, _ = cv2.projectPoints(points, rv, tv, scene[2], scene[3])
    pose = sol['pose_cube'](scene[1][0], points, pixels, K, dist)
    assert pose['rms'] < .8
    assert np.linalg.norm(pose['tvec'].ravel() - tv) < .02
    expected, _ = cv2.projectPoints(sol['CUBE'], rv, tv, scene[2], scene[3])
    actual, _ = cv2.projectPoints(sol['CUBE'], pose['rvec'], pose['tvec'], K, dist)
    assert np.max(np.linalg.norm(expected.reshape(-1, 2) - actual.reshape(-1, 2), axis=1)) < 2
