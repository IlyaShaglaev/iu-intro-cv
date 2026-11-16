import importlib.util
import io
import json
from pathlib import Path
import zipfile
import subprocess
import sys
import cv2

import numpy as np
import pytest

from cvcourse import datasets


def test_missing_checksum_prevents_optional_network(monkeypatch,tmp_path):
    monkeypatch.setattr(datasets,'CACHE',str(tmp_path))
    def forbidden(*args,**kwargs):
        raise AssertionError('Сеть не должна вызываться')
    monkeypatch.setattr(datasets,'_download_limited',forbidden)
    with pytest.raises(ValueError,match='PLACEHOLDER_MEASURE'):
        datasets.fetch('middlebury-piano')


def test_probe_does_not_treat_login_as_archive(monkeypatch):
    class Response:
        status = 200
        headers = {'Content-Type':'text/html'}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def geturl(self): return 'https://www.cvlibs.net/datasets/kitti/user_login.php'
    monkeypatch.setattr(datasets.urllib.request,'urlopen',lambda *a,**k: Response())
    report = datasets.probe('kitti-stereo-2015-10')
    assert report['state']=='registration_required'
    monkeypatch.setattr(Response,'geturl',lambda self: 'https://example.test/error.html')
    assert datasets.probe('middlebury-piano')['state']=='not_an_archive'


def test_download_limit_stops_before_body(monkeypatch,tmp_path):
    class Response:
        headers = {'Content-Length':'1000000001'}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,*args): raise AssertionError('Скачивание запрещено')
    monkeypatch.setattr(datasets.urllib.request,'urlopen',lambda *a,**k: Response())
    target = tmp_path/'download'
    with pytest.raises(ValueError,match='1 ГБ'):
        datasets._download_limited('https://example.test/archive.zip',str(target))
    assert not target.exists()


def test_import_kitti_selects_ten_and_hashes_files(tmp_path):
    archive = tmp_path/'local.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for i in range(11):
            for folder in ['image_2','image_3','disp_occ_0','disp_noc_0']:
                z.writestr('training/%s/%06d_10.png'%(folder,i), ('fixture-%s-%d'%(folder,i)).encode())
    destination = tmp_path/'selection'
    result = datasets.import_kitti(archive,destination,expected=datasets.sha256(archive))
    assert len(result['files'])==40
    assert not (destination/'training/image_2/000010_10.png').exists()
    manifest = json.loads((destination/'source-manifest.json').read_text(encoding='utf-8'))
    for path,digest in manifest['files'].items():
        assert digest==datasets.sha256(destination/path)
    with pytest.raises(ValueError,match='SHA-256'):
        datasets.import_kitti(archive,tmp_path/'bad',expected='0'*64)


def test_missing_kitti_member_leaves_no_selection(tmp_path):
    archive = tmp_path/'incomplete.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('training/image_2/000000_10.png',b'fixture')
    with pytest.raises(ValueError):
        datasets.import_kitti(archive,tmp_path/'selection')
    assert not (tmp_path/'selection').exists()


def test_external_pfm_units_and_vertical_orientation(tmp_path):
    path = Path(__file__).resolve().parents[1]/'homeworks/hw4-calib3d/template/prepare_external.py'
    spec = importlib.util.spec_from_file_location('external_adapter',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pixels = np.float32([[1,2],[3,4]])
    fixture = tmp_path/'disp.pfm'
    fixture.write_bytes(b'Pf\n2 2\n-1.0\n'+np.flipud(pixels).astype('<f4').tobytes())
    np.testing.assert_array_equal(module.pfm(fixture),pixels)


def test_downloader_no_arguments_never_fetches(monkeypatch):
    path = Path(__file__).resolve().parents[1]/'datasets/download.py'
    spec = importlib.util.spec_from_file_location('download_cli',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args,**kwargs): raise AssertionError('Неявная загрузка')
    monkeypatch.setattr(datasets,'fetch',forbidden)
    assert module.main([])==0


def test_middlebury_adapter_scaling_and_millimeters(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    image = np.zeros((96,128,3),np.uint8)
    for name in ['im0','im1']:
        assert cv2.imwrite(str(source/(name+'.png')),image)
    (source/'disp0.pfm').write_bytes(b'Pf\n128 96\n-1.0\n'+np.full((96,128),8,dtype='<f4').tobytes())
    (source/'calib.txt').write_text('cam0=[100 0 64; 0 100 48; 0 0 1]\nbaseline=120\ndoffs=2\n',encoding='utf-8')
    script = Path(__file__).resolve().parents[1]/'homeworks/hw4-calib3d/template/prepare_external.py'
    output = tmp_path/'prepared'
    proc = subprocess.run([sys.executable,'-B','-X','utf8',str(script),'middlebury',str(source),str(output),'--max-width','64'],
                          capture_output=True,text=True,encoding='utf-8')
    assert proc.returncode==0,proc.stderr
    with np.load(output/'gt.npz') as data:
        assert data['disparity'].shape==(48,64)
        np.testing.assert_allclose(data['disparity'],4)
        np.testing.assert_allclose(data['K'],[[50,0,31.75],[0,50,23.75],[0,0,1]])
        assert float(data['baseline_m'])==0.12 and float(data['doffs'])==1


def test_optional_checksum_mismatch_leaves_no_ready_cache(monkeypatch,tmp_path):
    monkeypatch.setattr(datasets,'CACHE',str(tmp_path/'cache'))
    def fake_download(url,destination):
        Path(destination).write_bytes(b'local format fixture, not an archive')
    monkeypatch.setattr(datasets,'_download_limited',fake_download)
    with pytest.raises(RuntimeError,match='сумма'):
        datasets.fetch('middlebury-piano',sha256_expected='0'*64)
    assert not (tmp_path/'cache/middlebury-piano').exists()
    assert not (tmp_path/'cache/middlebury-piano.download').exists()


def test_cached_optional_dataset_requires_matching_manifest(monkeypatch,tmp_path):
    monkeypatch.setattr(datasets,'CACHE',str(tmp_path))
    cache = tmp_path/'middlebury-piano'
    cache.mkdir()
    with pytest.raises(ValueError,match='манифест|manifest'):
        datasets.fetch('middlebury-piano')


def test_kitti_adapter_png_scale_and_calibration(tmp_path):
    source = tmp_path/'selection/training'
    for name in ['image_2','image_3','disp_occ_0','calib_cam_to_cam']:
        (source/name).mkdir(parents=True)
    for i in range(10):
        ident = '%06d'%i
        for name in ['image_2','image_3']:
            assert cv2.imwrite(str(source/name/(ident+'_10.png')),np.zeros((48,80,3),np.uint8))
        assert cv2.imwrite(str(source/'disp_occ_0'/(ident+'_10.png')),np.full((48,80),8*256,np.uint16))
        (source/'calib_cam_to_cam'/(ident+'.txt')).write_text(
            'P_rect_02: 100 0 40 0 0 100 24 0 0 0 1 0\n'
            'P_rect_03: 100 0 40 -12 0 100 24 0 0 0 1 0\n',encoding='utf-8')
    script = Path(__file__).resolve().parents[1]/'homeworks/hw4-calib3d/template/prepare_external.py'
    output = tmp_path/'prepared'
    proc = subprocess.run([sys.executable,'-B','-X','utf8',str(script),'kitti',str(source.parent),str(output)],
                          capture_output=True,text=True,encoding='utf-8')
    assert proc.returncode==0,proc.stderr
    assert len(list(output.glob('*/gt.npz')))==10
    with np.load(output/'000000/gt.npz') as data:
        np.testing.assert_allclose(data['disparity'],8)
        assert float(data['baseline_m'])==0.12
        assert float(data['doffs'])==0
