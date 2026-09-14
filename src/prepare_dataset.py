from pathlib import Path
from collections import defaultdict
import random

from PIL import Image


# ============================================================
# 1. ОБЩИЕ НАСТРОЙКИ
# ============================================================

# Фиксируем seed.
#
# random используется ниже для:
# - случайного выбора изображений;
# - перемешивания перед train/validation/test split.
#
# Благодаря фиксированному seed=42 при повторном запуске
# программа будет выбирать одни и те же изображения.
random.seed(42)


# Размер, к которому будем приводить изображения.
IMAGE_SIZE = (128, 128)


# Сколько исходных изображений хотим иметь в каждом целевом классе.
IMAGES_PER_CLASS = 1500


# Размеры частей одного класса:
#
# 1050 + 225 + 225 = 1500
TRAIN_COUNT = 1050
VALIDATION_COUNT = 225
TEST_COUNT = 225


# ============================================================
# 2. ПУТИ
# ============================================================

# __file__ — путь к текущему Python-файлу:
#
# PlantTinyML/src/prepare_dataset.py
#
# .parent -> src/
# .parent.parent -> PlantTinyML/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Здесь должен находиться распакованный исходный dataset.
#
# Если структура архива у тебя немного отличается,
# изменить нужно будет только эту строку.
RAW_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Plant Leaf Freshness and Disease Detection Dataset From Bangladesh"
    / "Original Dataset"
)


# Сюда программа создаст:
#
# prepared/
# ├── train/
# ├── validation/
# └── test/
PREPARED_DATASET_DIR = PROJECT_ROOT / "data" / "prepared"


# Какие расширения файлов считаем изображениями.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# 3. ПРОВЕРКА ПУТИ К DATASET
# ============================================================

if not RAW_DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Не найден исходный датасет:\n{RAW_DATASET_DIR}\n\n"
        "Проверь структуру папки data/raw."
    )


print("Исходный датасет:")
print(RAW_DATASET_DIR)
print()


# ============================================================
# 4. СОБИРАЕМ ПУТИ КО ВСЕМ ИЗОБРАЖЕНИЯМ
# ============================================================

# ВАЖНО:
#
# Здесь мы храним НЕ изображения в памяти.
# Мы храним Path — пути до файлов.
#
# Например:
#
# classes["tomato"] =
# [
#     Path(".../Tomato/Fresh/001.jpg"),
#     Path(".../Tomato/Disease/002.jpg"),
#     ...
# ]
#
classes = {
    "tomato": [],
    "cucumber": [],
    "other": [],
}


# Проходим по папкам верхнего уровня:
#
# Tomato/
# Cucumber/
# Eggplant/
# ...
for plant_folder in RAW_DATASET_DIR.iterdir():

    # Если встретился не каталог — пропускаем.
    if not plant_folder.is_dir():
        continue

    plant_name = plant_folder.name.lower()

    # Выполняем нашу собственную переразметку.
    #
    # Tomato -> tomato
    # Cucumber -> cucumber
    # всё остальное -> other
    if plant_name == "tomato":
        target_class = "tomato"

    elif plant_name == "cucumber":
        target_class = "cucumber"

    else:
        target_class = "other"

    # rglob("*") рекурсивно проходит по ВСЕМ подпапкам.
    #
    # Поэтому неважно, находятся изображения непосредственно
    # в Tomato/ или, например:
    #
    # Tomato/
    # ├── Fresh/
    # └── Some disease/
    #
    # Заболевание нас сейчас не интересует.
    for file_path in plant_folder.rglob("*"):

        if (
            file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            classes[target_class].append(file_path)


print("Количество изображений после объединения классов:")

for class_name, files in classes.items():
    print(f"{class_name:10s}: {len(files)}")

print()


# ============================================================
# 5. ВЫБИРАЕМ 1500 TOMATO И 1500 CUCUMBER
# ============================================================

selected = {
    "tomato": [],
    "cucumber": [],
    "other": [],
}


selected["tomato"] = random.sample(
    classes["tomato"],
    IMAGES_PER_CLASS,
)

selected["cucumber"] = random.sample(
    classes["cucumber"],
    IMAGES_PER_CLASS,
)


# ============================================================
# 6. ОТДЕЛЬНО БАЛАНСИРУЕМ OTHER
# ============================================================

# other состоит сразу из четырёх растений:
#
# Bitter Gourd
# Bottle gourd
# Cauliflower
# Eggplant
#
# Если просто случайно взять 1500 файлов из общего other,
# Eggplant может оказаться представлен сильнее остальных.
#
# Поэтому сначала снова группируем other
# по исходному виду растения.
other_by_plant = defaultdict(list)


for file_path in classes["other"]:

    # Допустим путь:
    #
    # .../Original Dataset/Eggplant/Fresh/abc.jpg
    #
    # relative_to(RAW_DATASET_DIR):
    #
    # Eggplant/Fresh/abc.jpg
    #
    # .parts:
    #
    # ("Eggplant", "Fresh", "abc.jpg")
    #
    # parts[0] -> "Eggplant"
    original_plant = file_path.relative_to(
        RAW_DATASET_DIR
    ).parts[0]

    other_by_plant[original_plant].append(file_path)


print("Состав исходного класса other:")

for plant_name, files in other_by_plant.items():
    print(f"{plant_name:15s}: {len(files)}")

print()


# У нас 4 исходных растения внутри other.
#
# 1500 / 4 = 375
OTHER_PER_PLANT = IMAGES_PER_CLASS // len(other_by_plant)


balanced_other = []


for plant_name, files in other_by_plant.items():

    chosen_files = random.sample(
        files,
        OTHER_PER_PLANT,
    )

    balanced_other.extend(chosen_files)


selected["other"] = balanced_other


print("После балансировки:")

for class_name, files in selected.items():
    print(f"{class_name:10s}: {len(files)}")

print()


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

splits = {
    "train": {
        "tomato": [],
        "cucumber": [],
        "other": [],
    },

    "validation": {
        "tomato": [],
        "cucumber": [],
        "other": [],
    },

    "test": {
        "tomato": [],
        "cucumber": [],
        "other": [],
    },
}


for class_name, files in selected.items():

    # Работаем с копией списка,
    # чтобы не менять selected.
    shuffled_files = files.copy()

    # Случайно перемешиваем.
    random.shuffle(shuffled_files)

    # Первые 1050 -> train
    splits["train"][class_name] = (
        shuffled_files[:TRAIN_COUNT]
    )

    # Следующие 225 -> validation
    val_start = TRAIN_COUNT
    val_end = TRAIN_COUNT + VALIDATION_COUNT

    splits["validation"][class_name] = (
        shuffled_files[val_start:val_end]
    )

    # Всё оставшееся -> test.
    #
    # Поскольку всего 1500:
    #
    # 1500 - 1050 - 225 = 225
    splits["test"][class_name] = (
        shuffled_files[val_end:]
    )


# ============================================================
# 8. ФУНКЦИЯ PREPROCESSING ИЗОБРАЖЕНИЯ
# ============================================================

def prepare_image(image_path: Path) -> Image.Image:
    """
    Загружает изображение и приводит его к формату,
    который позднее будет получать нейронная сеть.

    Результат:
        RGB
        128 x 128
    """

    # Открываем изображение.
    with Image.open(image_path) as image:

        # Не все исходные картинки обязательно имеют RGB.
        #
        # Например, могут быть grayscale или RGBA.
        #
        # Для CNN мы хотим всегда:
        #
        # R + G + B = 3 канала.
        image = image.convert("RGB")

        width, height = image.size

        # Берём меньшую сторону.
        #
        # Например:
        #
        # width  = 800
        # height = 600
        #
        # side = 600
        side = min(width, height)

        # Находим координаты центрального квадрата.
        left = (width - side) // 2
        top = (height - side) // 2

        right = left + side
        bottom = top + side

        # Вырезаем квадрат.
        cropped = image.crop(
            (left, top, right, bottom)
        )

        # Уменьшаем квадрат до 128x128.
        #
        # LANCZOS — качественный алгоритм ресэмплинга
        # при уменьшении изображения.
        resized = cropped.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )

        return resized


# ============================================================
# 9. СОЗДАЁМ ПАПКИ И СОХРАНЯЕМ PREPARED DATASET
# ============================================================

for split_name, split_classes in splits.items():

    for class_name, files in split_classes.items():

        # Например:
        #
        # data/prepared/train/tomato/
        output_folder = (
            PREPARED_DATASET_DIR
            / split_name
            / class_name
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Обработка {split_name}/{class_name}: "
            f"{len(files)} изображений"
        )

        for index, source_file in enumerate(files):

            prepared_image = prepare_image(source_file)

            # Например:
            #
            # tomato_0000.jpg
            # tomato_0001.jpg
            filename = f"{class_name}_{index:04d}.jpg"

            destination = output_folder / filename

            prepared_image.save(
                destination,
                quality=95,
            )


print()
print("Подготовка изображений завершена.")
print()


# ============================================================
# 10. ФИНАЛЬНАЯ ПРОВЕРКА
# ============================================================

total_images = 0
bad_images = 0


for split_name in ("train", "validation", "test"):

    print(split_name.upper())

    split_total = 0

    for class_name in ("tomato", "cucumber", "other"):

        folder = (
            PREPARED_DATASET_DIR
            / split_name
            / class_name
        )

        files = [
            file
            for file in folder.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        class_count = len(files)

        split_total += class_count
        total_images += class_count

        # Проверяем каждое сохранённое изображение.
        for file in files:

            with Image.open(file) as image:

                if (
                    image.size != IMAGE_SIZE
                    or image.mode != "RGB"
                ):
                    bad_images += 1
                    print("Проблема:", file)

        print(
            f"    {class_name:10s}: "
            f"{class_count}"
        )

    print(f"    TOTAL     : {split_total}")
    print()


print("Всего изображений:", total_images)
print("Некорректных изображений:", bad_images)