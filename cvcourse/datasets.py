# -*- coding: utf-8 -*-
"""Датасеты курса: скачивание с кэшем и проверкой контрольной суммы.

Данные **не хранятся в репозитории** — только реестр с адресами и SHA-256.
Кэш общий на машину (``~/.cache/cvcourse``), поэтому повторное занятие или
второй ноутбук не качают то же самое заново.

    from cvcourse import datasets
    path = datasets.fetch("middlebury-motorcycle")

Список того, что вообще есть: ``datasets.available()``.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile

CACHE = os.environ.get("CVCOURSE_CACHE") or os.path.join(
    os.path.expanduser("~"), ".cache", "cvcourse")

#: Реестр. sha256=None означает «сумма ещё не зафиксирована» — такие записи
#: закрываются на неделе 0, до выдачи материалов студентам.
REGISTRY = {
    "middlebury-motorcycle": dict(
        url="https://vision.middlebury.edu/stereo/data/scenes2014/zip/Motorcycle-perfect.zip",
        sha256=None, kind="zip", used_in="L12, занятие 12, ДЗ 4",
        optional=True, checksum_status="PLACEHOLDER_MEASURE",
        note="HW4: Motorcycle; опционально, базовые тесты — sample_data"),
    "middlebury-piano": dict(
        url="https://vision.middlebury.edu/stereo/data/scenes2014/zip/Piano-perfect.zip",
        sha256=None, kind="zip", used_in="HW4, расширение", optional=True,
        checksum_status="PLACEHOLDER_MEASURE", note="Middlebury 2014 Piano perfect"),
    "middlebury-pipes": dict(
        url="https://vision.middlebury.edu/stereo/data/scenes2014/zip/Pipes-perfect.zip",
        sha256=None, kind="zip", used_in="HW4, расширение", optional=True,
        checksum_status="PLACEHOLDER_MEASURE", note="Middlebury 2014 Pipes perfect"),
    "kitti-stereo-2015-10": dict(
        url=None, sha256=None, kind="dir", used_in="HW4, расширение", optional=True,
        checksum_status="PLACEHOLDER_MEASURE",
        access_url="https://www.cvlibs.net/datasets/kitti/user_login.php",
        page="https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo",
        note="Stereo 2015 training 000000_10–000009_10; локальный импорт, регистрация только для расширения"),
    "sintel-clean-alley": dict(
        url="http://files.is.tue.mpg.de/sintel/MPI-Sintel-testing.zip",
        sha256=None, kind="zip", used_in="L13, занятие 13",
        note="Оптический поток с ground truth, метрика EPE"),
    "cdnet-highway": dict(
        url="http://jacarini.dinf.usherbrooke.ca/static/dataset/baseline/highway.zip",
        sha256=None, kind="zip", used_in="занятие 13",
        note="Вычитание фона, готовая разметка переднего плана"),
    "oxford-buildings-lite": dict(
        url=None, sha256=None, kind="dir", used_in="занятие 14",
        note="Урезанная коллекция для мини-поиска BoW; собирается вручную, см. data/README.md"),
    "oxford-affine-graf": dict(
        url="https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz",
        sha256=None, kind="tar", used_in="занятия 8 и 9",
        note="Шесть видов одной сцены с известными гомографиями — repeatability"),
    "kitti-odometry-00": dict(
        url=None, sha256=None, kind="dir", used_in="L15, обзорный блок по VO",
        note="Только для демонстрации на лекции; регистрация на сайте KITTI"),
}


def available():
    """Печатает реестр: что есть, где используется, скачано ли."""
    for name, e in sorted(REGISTRY.items()):
        p = os.path.join(CACHE, name)
        state = "скачан" if os.path.exists(p) else "нет"
        print("%-26s %-8s %-24s %s" % (name, state, e["used_in"], e["note"]))


def fetch(name, *, force=False, quiet=False, sha256_expected=None):
    """Возвращает путь к каталогу датасета, скачав и распаковав при необходимости."""
    if name not in REGISTRY:
        raise KeyError("Неизвестный датасет %r. Доступные: %s"
                       % (name, ", ".join(sorted(REGISTRY))))
    e = REGISTRY[name]
    expected = sha256_expected or e['sha256']
    if expected and not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        raise ValueError('SHA-256 должна содержать 64 hex-символа')
    dst = os.path.join(CACHE, name)
    if os.path.exists(dst) and not force:
        if e.get('optional') and e['url']:
            manifest_path = os.path.join(dst, 'source-manifest.json')
            if not os.path.isfile(manifest_path):
                raise ValueError('Кэш расширения не имеет source-manifest.json; используйте force=True для проверки')
            with open(manifest_path, encoding='utf-8') as f:
                cached = json.load(f)
            if expected and cached.get('archive_sha256','').lower() != expected.lower():
                raise ValueError('Ожидаемая SHA-256 отличается от манифеста кэша')
        return dst
    if not e["url"]:
        raise RuntimeError(
            "Датасет %r нельзя скачать автоматически: %s\n  Инструкция — в "
            "data/README.md соответствующего занятия." % (name, e["note"]))

    if e.get('optional') and not expected:
        raise ValueError('Сумма PLACEHOLDER_MEASURE: передайте проверенную --sha256. '
                         'Базовые тесты HW4 используют sample_data без загрузок.')
    _mkdir(CACHE)
    tmp = dst + ".download"
    if e.get('optional'):
        _download_limited(e['url'], tmp)
    else:
        _download(e["url"], tmp, quiet=quiet)
    if expected:
        got = sha256(tmp)
        if got.lower() != expected.lower():
            os.remove(tmp)
            raise RuntimeError(
                "Контрольная сумма не сошлась для %s\n  ожидалась %s\n  получена %s\n"
                "  Скачивание оборвалось или файл на сервере изменился."
                % (name, expected, got))
    elif not quiet:
        sys.stderr.write("  ! sha256 для %s ещё не зафиксирована\n" % name)

    _mkdir(dst)
    try:
        _extract(tmp, dst, e["kind"])
        if e.get('optional'):
            with open(os.path.join(dst, 'source-manifest.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(dataset=name, url=e['url'], archive_sha256=got,
                               bytes=os.path.getsize(tmp)), f, indent=2)
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst


def sha256(path, *, chunk=1 << 20):
    """SHA-256 файла. Ей же фиксируются суммы в реестре."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe(name, timeout=15):
    """HEAD без загрузки; доступность URL не доказывает доступность архива за login."""
    e = REGISTRY[name]
    url = e['url'] or e.get('access_url')
    if not url:
        return dict(dataset=name, state='manual', url=None)
    try:
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'cvcourse/2026'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            final, content = response.geturl(), response.headers.get('Content-Type', '')
            state = 'registration_required' if e.get('access_url') or 'login' in final.lower() else 'available'
            if e['url'] and 'text/html' in content:
                state = 'not_an_archive'
            return dict(dataset=name, state=state, status=response.status, url=url,
                        final_url=final, content_type=content,
                        content_length=response.headers.get('Content-Length'))
    except Exception as ex:
        return dict(dataset=name, state='unavailable', url=url, error=str(ex))


def _download_limited(url, path, limit=1_000_000_000):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            length = response.headers.get('Content-Length')
            if length and int(length) > limit:
                raise ValueError('Архив >1 ГБ: требуется локальная копия, автоматическая загрузка остановлена')
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError('Сервер вернул HTML вместо архива')
            total = 0
            with open(path, 'wb') as output:
                while True:
                    block = response.read(min(1 << 20, limit-total+1))
                    if not block:
                        break
                    total += len(block)
                    if total > limit:
                        raise ValueError('Лимит 1 ГБ превышен')
                    output.write(block)
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise


def import_kitti(archive, destination, expected=None, calibration_archive=None):
    """Выбрать 10 пар training из локального архива; суммы вычисляются с файлов."""
    got = sha256(archive)
    if expected and got.lower() != expected.lower():
        raise ValueError('SHA-256 архива KITTI не совпадает')
    members = ['training/%s/%06d_10.png' % (folder, i) for i in range(10)
               for folder in ('image_2','image_3','disp_occ_0','disp_noc_0')]
    sources = [(archive, members)]
    if calibration_archive:
        sources.append((calibration_archive, ['training/calib_cam_to_cam/%06d.txt'%i for i in range(10)]))
    for source, wanted in sources:
        with zipfile.ZipFile(source) as z:
            missing = set(wanted) - set(z.namelist())
            if missing:
                raise ValueError('В архиве нет: ' + ', '.join(sorted(missing)[:3]))
    os.makedirs(destination, exist_ok=True)
    manifest = dict(dataset='kitti-stereo-2015-10', archive_sha256=got, files={})
    for source, wanted in sources:
        with zipfile.ZipFile(source) as z:
            for name in wanted:
                target = os.path.join(destination, *name.split('/'))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(name) as inp, open(target, 'wb') as out:
                    shutil.copyfileobj(inp, out)
                manifest['files'][name] = sha256(target)
    if calibration_archive:
        manifest['calibration_archive_sha256'] = sha256(calibration_archive)
    with open(os.path.join(destination, 'source-manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def clear(name=None):
    """Удаляет кэш целиком или один датасет."""
    target = CACHE if name is None else os.path.join(CACHE, name)
    shutil.rmtree(target, ignore_errors=True)
    return target


# ------------------------------------------------------------ внутреннее ---
def _mkdir(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def _download(url, dst, *, quiet=False):
    def hook(blocks, size, total):
        if quiet or total <= 0:
            return
        done = min(100, 100 * blocks * size // total)
        sys.stderr.write("\r  скачивание %d%% (%.1f МБ)" % (done, total / 1e6))
        sys.stderr.flush()
    try:
        urllib.request.urlretrieve(url, dst, hook)
    except Exception as ex:
        raise RuntimeError(
            "Не удалось скачать %s\n  %s: %s\n  Скачайте вручную и положите в %s"
            % (url, type(ex).__name__, ex, os.path.dirname(dst)))
    if not quiet:
        sys.stderr.write("\r  скачано: %s\n" % os.path.basename(dst))


def _extract(src, dst, kind):
    if kind == "zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(dst)
    elif kind == "tar":
        with tarfile.open(src) as t:
            t.extractall(dst)          # noqa: S202 — источники в реестре доверенные
    else:
        shutil.copy(src, dst)
