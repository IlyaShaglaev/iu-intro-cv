# -*- coding: utf-8 -*-
"""Опционально: привести локальный Middlebury/KITTI к формату команды stereo.

Ничего не скачивает. Исходные пары уже ректифицированы: повторная ректификация
испортила бы соответствие GT. Единицы: K в пикселях, baseline_m в метрах.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def pfm(path):
    with open(path, 'rb') as f:
        if f.readline().strip() != b'Pf':
            raise ValueError('Нужен одноканальный Pf PFM')
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        width,height = map(int,line.split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(),dtype='<f4' if scale < 0 else '>f4')
    return np.flipud(data.reshape(height,width)).astype(np.float32) * abs(scale)


def image(path, flags=cv2.IMREAD_COLOR):
    result = cv2.imdecode(np.fromfile(path,np.uint8),flags)
    if result is None:
        raise ValueError('Не прочитан файл '+str(path))
    return result


def save(out,left,right,disp,K,baseline,doffs,max_width,source):
    if left.shape != right.shape or disp.shape != left.shape[:2]:
        raise ValueError('Размеры GT/пары не совпадают')
    scale = min(1.,max_width/left.shape[1])
    if scale < 1:
        size = (round(left.shape[1]*scale),round(left.shape[0]*scale))
        # Фактический горизонтальный масштаб с учётом округления.
        sx,sy = size[0]/left.shape[1],size[1]/left.shape[0]
        left = cv2.resize(left,size,interpolation=cv2.INTER_AREA)
        right = cv2.resize(right,size,interpolation=cv2.INTER_AREA)
        disp = cv2.resize(disp,size,interpolation=cv2.INTER_NEAREST)*sx
        K = K.copy()
        K[0] *= sx
        K[1] *= sy
        # cv2.resize использует координаты центров: u'=(u+0.5)*sx-0.5.
        K[0,2] += (sx-1)/2
        K[1,2] += (sy-1)/2
        doffs *= sx
    valid = np.isfinite(disp) & (disp > 0)
    out.mkdir(parents=True,exist_ok=True)
    for name,img in [('left',left),('right',right)]:
        ok,encoded = cv2.imencode('.png',img)
        assert ok
        encoded.tofile(out/(name+'.png'))
    np.savez_compressed(out/'gt.npz',disparity=disp,valid=valid,K=K,
                        baseline_m=baseline,doffs=doffs)
    (out/'source.json').write_text(json.dumps(dict(source=source,rectified=True,
        image_size=[left.shape[1],left.shape[0]],baseline_m=baseline,doffs=float(doffs)),indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['middlebury','kitti'])
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--max-width',type=int,default=768)
    args = parser.parse_args()
    if args.max_width < 64:
        parser.error('--max-width >=64')
    if args.dataset=='middlebury':
        folder = args.source
        values = dict(line.split('=',1) for line in (folder/'calib.txt').read_text(encoding='utf-8').splitlines() if '=' in line)
        K = np.float64([float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',values['cam0'])]).reshape(3,3)
        save(args.output,image(folder/'im0.png'),image(folder/'im1.png'),pfm(folder/'disp0.pfm'),
             K,float(values['baseline'])/1000,float(values['doffs']),args.max_width,str(folder))
    else:
        root = args.source/'training' if (args.source/'training').exists() else args.source
        for i in range(10):
            ident = '%06d'%i
            path = root/'calib_cam_to_cam'/(ident+'.txt')
            if not path.exists():
                raise ValueError('Для метрической глубины нужны calibration files: '+str(path))
            values = dict(line.split(':',1) for line in path.read_text(encoding='utf-8').splitlines() if ':' in line)
            p2 = np.fromstring(values.get('P_rect_02',values.get('P2','')),sep=' ').reshape(3,4)
            p3 = np.fromstring(values.get('P_rect_03',values.get('P3','')),sep=' ').reshape(3,4)
            baseline = abs(p2[0,3]/p2[0,0]-p3[0,3]/p3[0,0])
            gt = image(root/'disp_occ_0'/(ident+'_10.png'),cv2.IMREAD_UNCHANGED).astype(np.float32)/256
            save(args.output/ident,image(root/'image_2'/(ident+'_10.png')),
                 image(root/'image_3'/(ident+'_10.png')),gt,p2[:,:3],baseline,
                 p3[0,2]-p2[0,2],args.max_width,str(root/ident))
    print('Готово:',args.output)


if __name__ == '__main__':
    main()
