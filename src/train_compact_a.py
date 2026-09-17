from pathlib import Path

import tensorflow as tf
import keras
import matplotlib.pyplot as plt


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

SEED = 42

# Максимальное число эпох.
# EarlyStopping может закончить обучение раньше.
EPOCHS = 20


# Для воспроизводимости
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


# Создаём директории, если их ещё нет
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# Проверяем dataset
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

    # Validation не перемешиваем.
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
    print(f"{index} -> {class_name}")


# В нашем проекте ожидаем ровно 3 класса
if len(class_names) != 3:
    raise ValueError(
        f"Ожидалось 3 класса, найдено: {len(class_names)}"
    )


# ============================================================
# 6. PREFETCH
# ============================================================

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 7. COMPACT CNN A
# ============================================================
#
# Baseline:
#
# 16 -> 32 -> 64 -> GAP -> Dense(32) -> Dense(3)
#
# Compact A:
#
# 16 -> 32 -> 64 -> GAP -> Dense(3)
#
# То есть мы убираем Dense(32), но сохраняем
# convolutional backbone baseline.
#
# Так мы проверяем конкретную гипотезу:
#
# нужен ли промежуточный Dense-слой после GAP?
# ============================================================

model_a = keras.Sequential(
    [

        # ------------------------------------------------------
        # INPUT
        # ------------------------------------------------------

        keras.layers.Input(
            shape=(128, 128, 3)
        ),


        # ------------------------------------------------------
        # NORMALIZATION
        # ------------------------------------------------------
        #
        # Из:
        # 0 ... 255
        #
        # В:
        # 0 ... 1
        # ------------------------------------------------------

        keras.layers.Rescaling(
            1.0 / 255
        ),


        # ------------------------------------------------------
        # CONV BLOCK 1
        # ------------------------------------------------------
        #
        # Input:
        # 128 x 128 x 3
        #
        # Output Conv:
        # 128 x 128 x 16
        #
        # После MaxPooling:
        # 64 x 64 x 16
        # ------------------------------------------------------

        keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),

            padding="same",

            activation="relu",
        ),

        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ------------------------------------------------------
        # CONV BLOCK 2
        # ------------------------------------------------------
        #
        # Input:
        # 64 x 64 x 16
        #
        # Output Conv:
        # 64 x 64 x 32
        #
        # После MaxPooling:
        # 32 x 32 x 32
        # ------------------------------------------------------

        keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),

            padding="same",

            activation="relu",
        ),

        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ------------------------------------------------------
        # CONV BLOCK 3
        # ------------------------------------------------------
        #
        # Input:
        # 32 x 32 x 32
        #
        # Output Conv:
        # 32 x 32 x 64
        #
        # После MaxPooling:
        # 16 x 16 x 64
        # ------------------------------------------------------

        keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),

            padding="same",

            activation="relu",
        ),

        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ------------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ------------------------------------------------------
        #
        # Было:
        #
        # 16 x 16 x 64
        #
        # Для каждого из 64 feature maps
        # берётся среднее значение.
        #
        # Получаем:
        #
        # 64 числа
        #
        # Output:
        # (64,)
        # ------------------------------------------------------

        keras.layers.GlobalAveragePooling2D(),


        # ------------------------------------------------------
        # CLASSIFIER
        # ------------------------------------------------------
        #
        # В baseline здесь было:
        #
        # Dense(32)
        # Dense(3)
        #
        # Сейчас сразу:
        #
        # 64 -> 3
        #
        # Это и есть главное изменение Compact A.
        # ------------------------------------------------------

        keras.layers.Dense(
            units=3,
            activation="softmax",
        ),
    ]
)


# ============================================================
# 8. SUMMARY
# ============================================================

print()
print("=" * 60)
print("COMPACT A")
print("=" * 60)

model_a.summary()

print()
print(
    "Количество параметров:",
    model_a.count_params()
)


# ============================================================
# 9. COMPILE
# ============================================================

model_a.compile(

    optimizer=keras.optimizers.Adam(),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ],
)


# ============================================================
# 10. CALLBACKS
# ============================================================

BEST_MODEL_PATH = (
    MODELS_DIR /
    "compact_a_best.keras"
)


# ------------------------------------------------------------
# ModelCheckpoint
# ------------------------------------------------------------
#
# После каждой эпохи смотрим validation accuracy.
#
# Если она стала лучше предыдущего рекорда,
# Keras сохраняет модель.
#
# Поэтому compact_a_best.keras всегда содержит
# лучшие веса, а не просто веса последней эпохи.
# ------------------------------------------------------------

checkpoint_callback = keras.callbacks.ModelCheckpoint(

    filepath=BEST_MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1,
)


# ------------------------------------------------------------
# EarlyStopping
# ------------------------------------------------------------
#
# Если validation accuracy не улучшается
# 3 эпохи подряд — прекращаем обучение.
#
# restore_best_weights=True означает:
# после остановки model_a тоже получит
# лучшие найденные веса.
# ------------------------------------------------------------

early_stopping_callback = keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=3,

    restore_best_weights=True,

    verbose=1,
)


# ============================================================
# 11. ОБУЧЕНИЕ
# ============================================================

print()
print("=" * 60)
print("НАЧИНАЕМ ОБУЧЕНИЕ COMPACT A")
print("=" * 60)

history_a = model_a.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
    ],
)


# ============================================================
# 12. ИЗВЛЕКАЕМ ИСТОРИЮ ОБУЧЕНИЯ
# ============================================================

train_accuracy = history_a.history["accuracy"]
val_accuracy = history_a.history["val_accuracy"]

train_loss = history_a.history["loss"]
val_loss = history_a.history["val_loss"]


# EarlyStopping может остановить обучение раньше,
# поэтому нельзя использовать range(1, EPOCHS + 1).

actual_epochs = len(
    history_a.history["loss"]
)

epochs_range = range(
    1,
    actual_epochs + 1
)


# ============================================================
# 13. ЛУЧШИЕ VALIDATION-МЕТРИКИ
# ============================================================

best_val_accuracy = max(
    val_accuracy
)

best_val_loss = min(
    val_loss
)


# Номер эпохи с лучшей accuracy
best_epoch = (
    val_accuracy.index(best_val_accuracy)
    + 1
)


print()
print("=" * 60)
print("РЕЗУЛЬТАТ COMPACT A")
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

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Количество параметров: "
    f"{model_a.count_params()}"
)


# ============================================================
# 14. ГРАФИК ACCURACY
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

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Compact A: Training and validation accuracy"
)

plt.legend()

plt.tight_layout()


ACCURACY_PLOT_PATH = (
    PLOTS_DIR /
    "compact_a_accuracy.png"
)

plt.savefig(
    ACCURACY_PLOT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 15. ГРАФИК LOSS
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

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Compact A: Training and validation loss"
)

plt.legend()

plt.tight_layout()


LOSS_PLOT_PATH = (
    PLOTS_DIR /
    "compact_a_loss.png"
)

plt.savefig(
    LOSS_PLOT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 16. ЗАГРУЖАЕМ ИМЕННО ЛУЧШУЮ МОДЕЛЬ
# ============================================================

best_model_a = keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# 17. ФИНАЛЬНАЯ VALIDATION-ПРОВЕРКА
# ============================================================

print()
print("=" * 60)
print("ПРОВЕРЯЕМ СОХРАНЁННУЮ BEST MODEL")
print("=" * 60)

validation_loss, validation_accuracy = (
    best_model_a.evaluate(
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
# 18. РАЗМЕР МОДЕЛИ НА ДИСКЕ
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
# 19. СОХРАНЯЕМ МЕТРИКИ В TXT
# ============================================================

METRICS_PATH = (
    METRICS_DIR /
    "compact_a_validation_metrics.txt"
)


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Compact A\n"
    )

    file.write(
        "==============================\n"
    )

    file.write(
        f"Classes: {class_names}\n"
    )

    file.write(
        f"Parameters: "
        f"{best_model_a.count_params()}\n"
    )

    file.write(
        f"Best epoch: "
        f"{best_epoch}\n"
    )

    file.write(
        f"Best val accuracy during training: "
        f"{best_val_accuracy:.6f}\n"
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
# 20. ИТОГ
# ============================================================

print()
print("=" * 60)
print("COMPACT A ГОТОВА")
print("=" * 60)

print()

print(
    "Best модель:"
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