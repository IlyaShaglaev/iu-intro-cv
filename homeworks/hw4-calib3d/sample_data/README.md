# HW4 — автономные данные и опциональные реальные наборы

## Три компактных примера курса

`case1/`, `case2/`, `case3/` (seed 10/11/12) созданы локально средствами
NumPy/OpenCV; это собственные синтетические данные, не изображения из KITTI.
Каждый пример содержит:

- `boards/`: 20 рендеров шахматки, 7×5 внутренних углов, клетка 0.03 м;
- `observations.npz`: объектные/наблюдаемые точки 20 поз и размер 640×480;
  к точкам добавлен шум σ=0.03 px, независимо от дискретизации рендеров;
- `measure.json`: калибровка/поза/проекции известного отрезка длиной 0.12 м;
- `left.png`, `right.png`: ректифицированная пара 256×96, известный сдвиг;
- `gt.npz`: disparity в пикселях, valid-mask, K, baseline_m, doffs;
- `scene.json`: истинные параметры синтетики; `expected.json`: измеренный вывод.

Измерено в окружении курса OpenCV 4.14.0.94:

| Пример | RMS по observations px | RMS по детекции boards px | SGBM bad2 % | RMSE px | Coverage % |
|---|---:|---:|---:|---:|---:|
| case1 | 0.040881 | 0.270475 | 0 | 0.007611 | 100 |
| case2 | 0.041017 | 0.286974 | 0 | 0.006902 | 100 |
| case3 | 0.041018 | 0.282186 | 0 | 0.007436 | 100 |

В каждом рендер-наборе нашлись все 20 поз. Измеренный отрезок по точным проекциям —
0.1200000000000001 м. Допуски тестов на синтетике: RMS observations <0.2 px,
элементы K не дальше 5 px от истины, bad2 <5 %, coverage >95 %, полный RMSE <1 px;
для точного пересечения лучей с плоскостью — координаты до 1e-8 м.
Это допуски контролируемой синтетики, не универсальные пороги качества своей камеры.
Рендеры имеют дискретизированные углы: их RMS закономерно выше, чем у точек NPZ.

SHA-256 входов реально вычислены и записаны в `manifest.json`.
Базовые тесты и CI используют только эти данные, без вызова download.py.
Установленные зависимости нужны; скачивание пакетов pip не относится к данным HW4.

## Что было заложено в курс

TA course-plan §4.4 задаёт Middlebury 2014 (3 сцены) и KITTI 2015 (10 пар).
L12, с. 53 PDF, вводит StereoSGBM и диспаратность; точные количества находятся
в плане курса, а не в извлекаемом тексте L11/L12. Исторический загрузчик обращался
к общему кэшу `~/.cache/cvcourse` (или CVCOURSE_CACHE): студент запускал его
самостоятельно. До этой выдачи реестр содержал только Motorcycle и KITTI odometry,
который **не является** KITTI Stereo 2015.

Теперь объём 3+10 сохранён как **опциональное расширение**. Нет требования
регистрироваться или скачивать архив ради базовых тестов и полного базового балла.

## Ссылки и выбранные данные

Официальные страницы: [Middlebury 2014](https://vision.middlebury.edu/stereo/data/scenes2014/),
[KITTI Stereo 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).

| Запись загрузчика | Что выбрано | HEAD 16.11.2026 |
|---|---|---|
| middlebury-motorcycle | Motorcycle-perfect | HTTP 200, application/zip, 105123861 байт |
| middlebury-piano | Piano-perfect | HTTP 200, application/zip, 94563379 байт |
| middlebury-pipes | Pipes-perfect | HTTP 200, application/zip, 103057103 байт |
| kitti-stereo-2015-10 | training 000000_10 … 000009_10 | Страница входа HTTP 200; требуется регистрация для архива |

Это проверка HTTP-заголовков, не скачивание архивов и не вычисление их SHA-256.
Суммы архивов — `PLACEHOLDER_MEASURE`, пока нет фактически полученных файлов.
По официальной странице архив KITTI обозначен как 2 ГБ; автоматически он
не скачивается. Доступ к странице входа не означает доступ к данным.

## Воспроизводимая опциональная загрузка

Из корня `iu-intro-cv`:

```powershell
python datasets/download.py --list
python datasets/download.py --check middlebury-motorcycle middlebury-piano middlebury-pipes kitti-stereo-2015-10
python datasets/download.py middlebury-motorcycle --sha256 <проверенная_SHA256_архива>
```

Повторите последнюю команду для piano/pipes с соответствующей суммой. Ожидаемую
сумму получайте из доверенного манифеста или реально скачанного локального архива;
не заменяйте PLACEHOLDER_MEASURE случайной строкой. Загрузчик сравнит её с байтами
файла и сохранит `source-manifest.json`. Без суммы загрузка расширения остановится
до сети. Без аргументов CLI показывает справку. Потоки >1 ГБ остановятся.

Для имеющейся локальной разрешённой копии официального KITTI training:

```powershell
python datasets/download.py --import-kitti D:/datasets/data_scene_flow.zip --calibration-archive D:/datasets/data_scene_flow_calib.zip --dest D:/datasets/kitti-stereo-2015-10
```

Выбираются ровно 10 идентификаторов, для каждого image_2/image_3/disp_occ_0/disp_noc_0
и calibration-файл. `--sha256` дополнительно проверяет локальный основной архив.
Манифест хранит измеренные суммы архивов и выбранных файлов. Архивы не копируются
в репозиторий, интернет для импорта не нужен. Без calibration-архива пары можно
импортировать для диспаратности, но метрическую глубину вычислять нельзя.

## Приведение к формату команды stereo

В `template/` (или скопированном проекте):

```powershell
python prepare_external.py middlebury D:/datasets/Motorcycle-perfect external/motorcycle
python prepare_external.py kitti D:/datasets/kitti-stereo-2015-10 external/kitti
python calib3d.py stereo external/motorcycle --num-disparities 128 -o out/motorcycle
```

Адаптер ничего не скачивает. Middlebury читает disp0.pfm и calib.txt: baseline
из мм → м, doffs учтён в Z. KITTI читает uint16 GT /256, calibration P_rect_02/03
(либо P2/P3). При уменьшении пары корректируются K, disparity и doffs;
GT ресайзится nearest-neighbor, маска строится из конечных положительных значений.
Выбранные пары уже ректифицированы; повторно выпрямлять их без преобразования GT нельзя.
Параметры numDisparities подбираются под масштаб и диапазон, 32 — только старт синтетики.
bad2 курса не тождественна официальной KITTI D1 (3 px/5 %); в отчёте не смешивайте названия.
