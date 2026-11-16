# -*- coding: utf-8 -*-
"""Скачать датасеты курса одной командой.

    python datasets/download.py                  # справка; ничего не скачивает
    python datasets/download.py middlebury-motorcycle sintel-clean-alley
    python datasets/download.py --list

Кэш общий на машину: ``~/.cache/cvcourse`` либо ``$CVCOURSE_CACHE``.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvcourse import datasets


def main(argv):
    parser = argparse.ArgumentParser(description='Датасеты курса; HW4 работает офлайн на sample_data')
    parser.add_argument('names', nargs='*')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--check', action='store_true', help='HEAD без загрузки архивов')
    parser.add_argument('--sha256', help='Ожидаемая SHA-256 одного архива')
    parser.add_argument('--import-kitti', metavar='ZIP', help='Локальный KITTI Stereo 2015 training')
    parser.add_argument('--calibration-archive', metavar='ZIP')
    parser.add_argument('--dest', help='Каталог импорта KITTI')
    args = parser.parse_args(argv)
    if args.list:
        datasets.available()
        return 0

    if args.import_kitti:
        try:
            dest = args.dest or os.path.join(datasets.CACHE, 'kitti-stereo-2015-10')
            result = datasets.import_kitti(args.import_kitti, dest, args.sha256, args.calibration_archive)
            print(json.dumps(dict(destination=dest, files=len(result['files']),
                                  archive_sha256=result['archive_sha256'])))
            return 0
        except Exception as ex:
            print(str(ex), file=sys.stderr)
            return 1
    if not args.names:
        parser.print_help()
        return 0
    if args.sha256 and len(args.names) != 1:
        parser.error('--sha256 применяется ровно к одному архиву')

    failed = []
    for n in args.names:
        print("==", n)
        try:
            if args.check:
                report = datasets.probe(n)
                print(json.dumps(report, ensure_ascii=False))
                if report['state'] in ('unavailable', 'not_an_archive'):
                    failed.append(n)
            else:
                print("   ->", datasets.fetch(n, sha256_expected=args.sha256))
        except Exception as e:                      # noqa: BLE001
            print("   !!", e)
            failed.append(n)

    if failed:
        print("\nНе скачано / не проверено: %s" % ", ".join(failed))
        print("Инструкции по ручной установке — в data/README.md нужного занятия.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
