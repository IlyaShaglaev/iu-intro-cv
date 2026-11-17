import importlib.util
import os
import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Базовые тесты ДЗ не должны обращаться в сеть')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)


@pytest.fixture(scope='session')
def hw():
    path = Path(os.environ.get('HW_SOLUTION', Path(__file__).resolve().parents[1]/'calib3d.py'))
    spec = importlib.util.spec_from_file_location('hw_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def samples():
    root = Path(__file__).resolve().parents[1]
    default = root/'sample_data' if (root/'sample_data').exists() else root.parent/'sample_data'
    return Path(os.environ.get('HW_SAMPLE_DATA', default))
