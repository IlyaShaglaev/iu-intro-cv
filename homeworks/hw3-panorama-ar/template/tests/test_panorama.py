import json
import cv2
import numpy as np
import pytest


@pytest.mark.parametrize('case', ['case1','case2','case3'])
def test_agreed_dlt_control(hw, samples, case):
    data = json.loads((samples/case/'correspondences.json').read_text(encoding='utf-8'))
    src,dst = np.float64(data['src']),np.float64(data['dst'])
    H = hw.dlt(src,dst)
    ref,_ = cv2.findHomography(src,dst,method=0)
    assert np.linalg.norm(H/H[2,2] - ref/ref[2,2]) < 1e-6
    np.testing.assert_allclose(hw.project(src,H),dst,atol=1e-9)


def test_hartley_large_coordinates(hw):
    p = np.float64([[1e6,2e6],[1e6+20,2e6],[1e6+10,2e6+30],[1e6-15,2e6+25]])
    q,T = hw.normalize_points(p)
    np.testing.assert_allclose(q.mean(0),0,atol=1e-9)
    assert abs(np.linalg.norm(q,axis=1).mean()-np.sqrt(2)) < 1e-9
    np.testing.assert_allclose(hw.project(p,T),q,atol=1e-9)


def test_projective_dlt(hw):
    src = np.float64([[0,0],[30,0],[30,20],[0,20],[8,6],[20,8],[9,18]])
    truth = np.float64([[1.1,0.08,5],[-0.04,0.95,3],[0.004,-0.002,1]])
    dst = cv2.perspectiveTransform(src[None],truth)[0]
    np.testing.assert_allclose(hw.dlt(src,dst),truth,atol=1e-10)


def test_dlt_collinear(hw):
    p = np.float64([[0,0],[1,1],[2,2],[3,3]])
    with pytest.raises(ValueError):
        hw.dlt(p,p)


def test_ransac_outliers(hw):
    rng = np.random.default_rng(83)
    src = rng.uniform(0,100,(70,2))
    truth = np.float64([[1,0.05,12],[-0.02,1,7],[0.0001,0,1]])
    dst = cv2.perspectiveTransform(src[None],truth)[0]
    dst[:20] = rng.uniform(-100,200,(20,2))
    H,mask = hw.ransac_homography(src,dst,threshold=0.05,iterations=250,seed=41)
    assert mask.dtype == bool and mask.sum() == 50 and not mask[:20].any()
    np.testing.assert_allclose(H,truth,atol=1e-8)


@pytest.mark.parametrize('detector', ['sift','orb'])
def test_features_and_geometry(hw,samples,detector):
    a,b = [hw.read_image(samples/'case1'/('frame%d.png'%i)) for i in [0,1]]
    p,q = hw.match_points(a,b,detector)
    assert len(p) >= 4
    H,mask = hw.ransac_homography(p,q,iterations=150)
    assert mask.mean() > 0.5
    assert abs(H[0,2]+120) < 1


def test_textureless_is_failure(hw):
    image = np.zeros((80,80,3),np.uint8)
    with pytest.raises(ValueError):
        hw.match_points(image,image)


def test_black_pixels_have_coverage(hw):
    black = np.zeros((12,16,3),np.uint8)
    white = np.full_like(black,200)
    image,shift = hw.feather_blend([black,white],[np.eye(3),np.eye(3)])
    np.testing.assert_allclose(image,100,atol=1)
    np.testing.assert_allclose(shift,np.eye(3))


def test_panorama_canvas_and_translation(hw,samples):
    folder = samples/'case1'
    images = [hw.read_image(folder/('frame%d.png'%i)) for i in range(4)]
    transforms = json.loads((folder/'scene.json').read_text(encoding='utf-8'))['transforms']
    image,_ = hw.feather_blend(images,[np.float64(H) for H in transforms])
    assert image.shape == (180,700,3)
    np.testing.assert_array_equal(image[:,:100],images[0][:,:100])
    np.testing.assert_array_equal(image[:,-100:],images[-1][:,-100:])


def test_smoothing_reduces_jump(hw):
    p = np.float64([[0,0],[20,0],[20,10],[0,10]])
    H = np.float64([[1,0,10],[0,1,6],[0,0,1]])
    result = hw.smooth_homography(np.eye(3),H,p,alpha=0.25)
    np.testing.assert_allclose(hw.project(p,result),p+[2.5,1.5],atol=1e-9)


def test_overlay_mask(hw):
    frame = np.full((20,30,3),100,np.uint8)
    texture = np.zeros((5,7,3),np.uint8)
    H = np.float64([[1,0,4],[0,1,6],[0,0,1]])
    out = hw.overlay(frame,texture,H,opacity=1)
    assert (out[6:11,4:11] == 0).all()
    assert (out[:6] == 100).all()
