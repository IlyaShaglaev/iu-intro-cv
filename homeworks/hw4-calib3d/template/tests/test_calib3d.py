import json
import cv2
import numpy as np
import pytest


def test_board_order_and_units(hw):
    points = hw.board_points(7,5,0.03)
    assert points.shape == (35,3)
    np.testing.assert_allclose(points[1],[0.03,0,0],atol=1e-8)
    np.testing.assert_allclose(points[7],[0,0.03,0],atol=1e-8)
    assert not points[:,2].any()


@pytest.mark.parametrize('case',['case1','case2','case3'])
def test_calibration_recovers_intrinsics(hw,samples,case):
    folder = samples/case
    with np.load(folder/'observations.npz',allow_pickle=False) as data:
        rms,K,dist,r,t,errors = hw.calibrate(data['object_points'],data['image_points'],data['image_size'])
    truth = np.float64(json.loads((folder/'scene.json').read_text(encoding='utf-8'))['K'])
    assert rms < 0.2 and len(errors)==20
    assert np.max(np.abs(K-truth)) < 5
    assert abs(rms-np.sqrt(np.mean(errors**2))) < 1e-3


def test_reprojection_is_euclidean_rms(hw):
    obj = np.float32([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    K = np.eye(3)
    r,t = np.zeros(3),np.float64([0,0,5])
    image,_ = cv2.projectPoints(obj,r,t,K,None)
    errors = hw.reprojection_errors([obj],[image+np.float32([3,4])],K,None,[r],[t])
    np.testing.assert_allclose(errors,[5],atol=1e-6)


def test_too_few_poses(hw):
    with pytest.raises(ValueError):
        hw.calibrate([],[],(640,480))


def test_planar_metrology(hw,samples):
    spec = json.loads((samples/'case1'/'measure.json').read_text(encoding='utf-8'))
    xyz = hw.plane_measure(spec['pixels'],np.float64(spec['K']),np.float64(spec['distortion']),
                           np.float64(spec['rvec']),np.float64(spec['tvec_m']))
    np.testing.assert_allclose(xyz,[[0.02,0.03,0],[0.14,0.03,0]],atol=1e-8)


def test_bad2_boundary_and_gt_mask(hw):
    truth = np.array([[8.,8.,8.,0.]])
    pred = np.array([[8.,10.,11.,999.]])
    report = hw.disparity_metrics(pred,truth,np.array([[1,1,1,0]],bool))
    assert report['valid_pixels']==3
    assert abs(report['bad2_pct']-100/3) < 1e-10
    assert abs(report['rmse_px']-np.sqrt(13/3)) < 1e-10


def test_invalid_predictions_are_not_dropped(hw):
    result = hw.disparity_metrics(np.array([np.nan,0.,-1.,8.]),np.full(4,8.),np.ones(4,bool))
    assert result['bad2_pct']==75 and result['coverage_pct']==25
    assert result['rmse_px'] is None


def test_empty_mask_fails(hw):
    with pytest.raises(ValueError):
        hw.disparity_metrics(np.ones(3),np.ones(3),np.zeros(3,bool))


def test_depth_units_and_offset(hw):
    z = hw.stereo_depth(np.array([8.,16.,0.,-1.,np.nan]),520,0.12,doffs=2)
    np.testing.assert_allclose(z[:2],[6.24,62.4/18])
    assert np.isnan(z[2:]).all()


def test_point_cloud_reprojects(hw):
    K = np.float64([[20,0,1],[0,30,1],[0,0,1]])
    depth = np.full((3,4),2.)
    xyz = hw.point_cloud(depth,K)
    np.testing.assert_allclose(xyz[1,1],[0,0,2])
    v,u = np.indices(depth.shape)
    np.testing.assert_allclose(xyz[...,0]/xyz[...,2]*20+1,u)
    np.testing.assert_allclose(xyz[...,1]/xyz[...,2]*30+1,v)


@pytest.mark.parametrize('case',['case1','case2','case3'])
def test_sgbm_pixel_units_and_depth(hw,samples,case):
    folder = samples/case
    disp = hw.sgbm(hw.read_image(folder/'left.png'),hw.read_image(folder/'right.png'))
    with np.load(folder/'gt.npz',allow_pickle=False) as data:
        result = hw.disparity_metrics(disp,data['disparity'],data['valid'])
    assert result['coverage_pct'] > 95
    assert result['bad2_pct'] < 5
    assert result['rmse_px'] is not None and result['rmse_px'] < 1


def test_sgbm_rejects_wrong_disparity_range(hw):
    with pytest.raises(ValueError):
        hw.sgbm(np.zeros((40,80),np.uint8),np.zeros((40,80),np.uint8),num_disparities=15)
