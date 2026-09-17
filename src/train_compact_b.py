from pathlib import Path

import tensorflow as tf
import keras

# На macOS используем неинтерактивный backend,
# чтобы plt.show() не ронял Python.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

SEED = 42

# Максимальное количество эпох.
# EarlyStopping завершит обучение раньше,
# если validation accuracy перестанет улучшаться.
EPOCHS = 20


# Фиксируем random seed для воспроизводимости.
keras.utils.set_random_seed(SEED)


# ============================================================
# 2. ПУТИ
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "prepared"

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Не найдена папка train: {TRAIN_DIR}"
    )

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Не найдена папка validation: {VALIDATION_DIR}"
    )


# ============================================================
# 3. ЗАГРУЖАЕМ TRAIN
# ============================================================

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,

    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=True,
    seed=SEED,

    color_mode="rgb",
)


# ============================================================
# 4. ЗАГРУЖАЕМ VALIDATION
# ============================================================

validation_ds = keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,

    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=False,

    color_mode="rgb",
)


# ============================================================
# 5. ПРОВЕРЯЕМ КЛАССЫ
# ============================================================

class_names = train_ds.class_names

print()
print("Классы:")

for index, class_name in enumerate(class_names):
    print(
        f"{index} -> {class_name}"
    )


if len(class_names) != 3:
    raise ValueError(
        f"Ожидалось 3 класса, найдено: {len(class_names)}"
    )


# ============================================================
# 6. ПРОВЕРЯЕМ ОДИН BATCH
# ============================================================

for images, labels in train_ds.take(1):

    print()

    print(
        "Форма изображений:",
        images.shape,
    )

    print(
        "Форма labels:",
        labels.shape,
    )

    print(
        "Тип изображений:",
        images.dtype,
    )

    print(
        "Минимальное значение пикселя:",
        tf.reduce_min(images).numpy(),
    )

    print(
        "Максимальное значение пикселя:",
        tf.reduce_max(images).numpy(),
    )

    print(
        "Первые labels:",
        labels[:10].numpy(),
    )


# ============================================================
# 7. PREFETCH
# ============================================================

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 8. СОЗДАЁМ COMPACT MODEL B
# ============================================================
#
# Архитектура:
#
# 128x128x3
#     ↓
# Conv2D(8)
#     ↓
# MaxPooling
#     ↓
# Conv2D(16)
#     ↓
# MaxPooling
#     ↓
# Conv2D(32)
#     ↓
# MaxPooling
#     ↓
# GlobalAveragePooling
#     ↓
# Dense(3, softmax)
#
# ============================================================

model = keras.Sequential(
    [

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        keras.layers.Input(
            shape=(128, 128, 3)
        ),


        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------
        #
        # Пиксели:
        #
        # 0 ... 255
        #
        # превращаются в:
        #
        # 0 ... 1
        # ----------------------------------------------------

        keras.layers.Rescaling(
            1.0 / 255
        ),


        # ----------------------------------------------------
        # CONV BLOCK 1
        # ----------------------------------------------------
        #
        # Input:
        #
        # 128 x 128 x 3
        #
        # Conv:
        #
        # 128 x 128 x 8
        #
        # Pool:
        #
        # 64 x 64 x 8
        # ----------------------------------------------------

        keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ----------------------------------------------------
        # CONV BLOCK 2
        # ----------------------------------------------------
        #
        # 64 x 64 x 8
        #
        # ↓
        #
        # 64 x 64 x 16
        #
        # ↓
        #
        # 32 x 32 x 16
        # ----------------------------------------------------

        keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ----------------------------------------------------
        # CONV BLOCK 3
        # ----------------------------------------------------
        #
        # 32 x 32 x 16
        #
        # ↓
        #
        # 32 x 32 x 32
        #
        # ↓
        #
        # 16 x 16 x 32
        # ----------------------------------------------------

        keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------
        #
        # Было:
        #
        # 16 x 16 x 32
        #
        # Для каждого из 32 feature maps
        # считаем среднее значение.
        #
        # Получаем вектор:
        #
        # 32
        # ----------------------------------------------------

        keras.layers.GlobalAveragePooling2D(),


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------
        #
        # 32 признака -> 3 класса
        #
        # cucumber
        # other
        # tomato
        # ----------------------------------------------------

        keras.layers.Dense(
            units=3,
            activation="softmax",
        ),
    ]
)


# ============================================================
# 9. SUMMARY
# ============================================================

print()
print("=" * 60)
print("COMPACT B")
print("=" * 60)

model.summary()

print()

print(
    "Количество параметров:",
    model.count_params(),
)


# ============================================================
# 10. COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ],
)


# ============================================================
# 11. CALLBACKS
# ============================================================

BEST_MODEL_PATH = (
    MODELS_DIR /
    "compact_b_best.keras"
)


# Сохраняем только лучший checkpoint
# по validation accuracy.
checkpoint_callback = (
    keras.callbacks.ModelCheckpoint(

        filepath=BEST_MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1,
    )
)


# Если validation accuracy не улучшается
# 3 эпохи подряд — останавливаем обучение.
early_stopping_callback = (
    keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=3,

        restore_best_weights=True,

        verbose=1,
    )
)


# ============================================================
# 12. ОБУЧЕНИЕ
# ============================================================

print()
print("=" * 60)
print("ОБУЧЕНИЕ COMPACT B")
print("=" * 60)


history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
    ],
)


# ============================================================
# 13. ИСТОРИЯ ОБУЧЕНИЯ
# ============================================================

train_accuracy = (
    history.history["accuracy"]
)

val_accuracy = (
    history.history["val_accuracy"]
)

train_loss = (
    history.history["loss"]
)

val_loss = (
    history.history["val_loss"]
)


# EarlyStopping может остановить обучение,
# например, на 16-й эпохе вместо 20-й.
actual_epochs = len(
    history.history["loss"]
)


epochs_range = range(
    1,
    actual_epochs + 1,
)


# ============================================================
# 14. ЛУЧШАЯ ЭПОХА
# ============================================================

best_val_accuracy = max(
    val_accuracy
)


best_epoch = (
    val_accuracy.index(
        best_val_accuracy
    )
    + 1
)


print()
print("=" * 60)
print("РЕЗУЛЬТАТ ОБУЧЕНИЯ")
print("=" * 60)

print(
    f"Лучший epoch: {best_epoch}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# 15. ГРАФИК ACCURACY
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs_range,
    train_accuracy,
    label="Train accuracy",
)


plt.plot(
    epochs_range,
    val_accuracy,
    label="Validation accuracy",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Compact B: training and validation accuracy"
)

plt.legend()

plt.tight_layout()


ACCURACY_PLOT_PATH = (
    PLOTS_DIR /
    "compact_b_accuracy.png"
)


plt.savefig(
    ACCURACY_PLOT_PATH,

    dpi=200,

    bbox_inches="tight",
)


# Освобождаем figure.
# plt.show() не используем.
plt.close()


# ============================================================
# 16. ГРАФИК LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs_range,
    train_loss,
    label="Train loss",
)


plt.plot(
    epochs_range,
    val_loss,
    label="Validation loss",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Compact B: training and validation loss"
)

plt.legend()

plt.tight_layout()


LOSS_PLOT_PATH = (
    PLOTS_DIR /
    "compact_b_loss.png"
)


plt.savefig(
    LOSS_PLOT_PATH,

    dpi=200,

    bbox_inches="tight",
)


plt.close()


# ============================================================
# 17. ЗАГРУЖАЕМ ЛУЧШИЙ CHECKPOINT
# ============================================================
#
# Не используем просто model после последней эпохи.
#
# Загружаем именно тот .keras-файл,
# который ModelCheckpoint сохранил
# при максимальной validation accuracy.
# ============================================================

best_model = keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# 18. ПРОВЕРЯЕМ BEST MODEL НА VALIDATION
# ============================================================

print()
print("=" * 60)
print("ПРОВЕРКА BEST COMPACT B")
print("=" * 60)


validation_loss, validation_accuracy = (
    best_model.evaluate(
        validation_ds,
        verbose=1,
    )
)


print()

print(
    f"Validation accuracy: "
    f"{validation_accuracy:.4f}"
)

print(
    f"Validation accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Validation loss: "
    f"{validation_loss:.4f}"
)


# ============================================================
# 19. РАЗМЕР .KERAS
# ============================================================

model_size_bytes = (
    BEST_MODEL_PATH.stat().st_size
)


model_size_kib = (
    model_size_bytes / 1024
)


print()

print(
    f"Размер .keras модели: "
    f"{model_size_bytes} bytes"
)

print(
    f"Размер .keras модели: "
    f"{model_size_kib:.2f} KiB"
)


# ============================================================
# 20. СОХРАНЯЕМ МЕТРИКИ
# ============================================================

METRICS_PATH = (
    METRICS_DIR /
    "compact_b_validation_metrics.txt"
)


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Compact B\n"
    )

    file.write(
        "==============================\n"
    )

    file.write(
        f"Classes: {class_names}\n"
    )

    file.write(
        f"Parameters: "
        f"{best_model.count_params()}\n"
    )

    file.write(
        f"Best epoch: "
        f"{best_epoch}\n"
    )

    file.write(
        f"Validation accuracy: "
        f"{validation_accuracy:.6f}\n"
    )

    file.write(
        f"Validation loss: "
        f"{validation_loss:.6f}\n"
    )

    file.write(
        f"Model size bytes: "
        f"{model_size_bytes}\n"
    )

    file.write(
        f"Model size KiB: "
        f"{model_size_kib:.2f}\n"
    )


# ============================================================
# 21. ИТОГ
# ============================================================

print()
print("=" * 60)
print("COMPACT B ГОТОВА")
print("=" * 60)

print()

print(
    "Best model:"
)

print(
    BEST_MODEL_PATH
)


print()

print(
    "Accuracy plot:"
)

print(
    ACCURACY_PLOT_PATH
)


print()

print(
    "Loss plot:"
)

print(
    LOSS_PLOT_PATH
)


print()

print(
    "Metrics:"
)

print(
    METRICS_PATH
)


print()

print(
    "TEST SET НЕ ИСПОЛЬЗОВАЛСЯ."
)