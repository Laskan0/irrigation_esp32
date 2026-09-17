from pathlib import Path

import tensorflow as tf
import keras

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
# EarlyStopping может остановить обучение раньше.
EPOCHS = 20


# Делаем эксперименты воспроизводимыми.
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
# 6. СМОТРИМ ОДИН BATCH
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
# 7. ПОКАЗЫВАЕМ НЕСКОЛЬКО ИЗОБРАЖЕНИЙ
# ============================================================

plt.figure(
    figsize=(8, 8)
)

for i in range(9):

    plt.subplot(
        3,
        3,
        i + 1,
    )

    plt.imshow(
        images[i]
        .numpy()
        .astype("uint8")
    )

    label_index = int(
        labels[i].numpy()
    )

    label_name = (
        class_names[label_index]
    )

    plt.title(
        label_name
    )

    plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 8. PREFETCH
# ============================================================

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 9. BASELINE MODEL
# ============================================================
#
# Архитектура:
#
# 16 -> 32 -> 64 -> GAP -> Dense(32) -> Dense(3)
#
# ============================================================

model = keras.Sequential(
    [

        keras.layers.Input(
            shape=(128, 128, 3)
        ),

        # Нормализация:
        # 0..255 -> 0..1
        keras.layers.Rescaling(
            1.0 / 255
        ),


        # ----------------------------------------------------
        # CONV BLOCK 1
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
        # CONV BLOCK 2
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
        # CONV BLOCK 3
        # ----------------------------------------------------

        keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # Было:
        #
        # 16 x 16 x 64
        #
        # Станет:
        #
        # 64 признака
        keras.layers.GlobalAveragePooling2D(),


        # Дополнительный классифицирующий слой
        keras.layers.Dense(
            units=32,
            activation="relu",
        ),


        # 3 класса
        keras.layers.Dense(
            units=3,
            activation="softmax",
        ),
    ]
)


# ============================================================
# 10. BASELINE SUMMARY
# ============================================================

print()
print("=" * 60)
print("BASELINE")
print("=" * 60)

model.summary()

print()

print(
    "Количество параметров baseline:",
    model.count_params(),
)


# ============================================================
# 11. COMPILE BASELINE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ],
)


# ============================================================
# 12. CALLBACKS BASELINE
# ============================================================

BASELINE_PATH = (
    MODELS_DIR /
    "baseline_best.keras"
)


baseline_checkpoint = (
    keras.callbacks.ModelCheckpoint(

        filepath=BASELINE_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1,
    )
)


baseline_early_stopping = (
    keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=3,

        restore_best_weights=True,

        verbose=1,
    )
)


# ============================================================
# 13. ОБУЧЕНИЕ BASELINE
# ============================================================

print()
print("=" * 60)
print("ОБУЧЕНИЕ BASELINE")
print("=" * 60)


history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=[
        baseline_checkpoint,
        baseline_early_stopping,
    ],
)


# ============================================================
# 14. МЕТРИКИ BASELINE
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


actual_epochs = len(
    history.history["loss"]
)


epochs_range = range(
    1,
    actual_epochs + 1,
)


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
print("BASELINE")

print(
    "Лучший epoch:",
    best_epoch,
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
# 15. ГРАФИК BASELINE ACCURACY
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
    "Baseline: training and validation accuracy"
)

plt.legend()

plt.tight_layout()


BASELINE_ACCURACY_PLOT = (
    PLOTS_DIR /
    "baseline_accuracy.png"
)


plt.savefig(
    BASELINE_ACCURACY_PLOT,

    dpi=200,

    bbox_inches="tight",
)


plt.close()


# ============================================================
# 16. ГРАФИК BASELINE LOSS
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
    "Baseline: training and validation loss"
)

plt.legend()

plt.tight_layout()


BASELINE_LOSS_PLOT = (
    PLOTS_DIR /
    "baseline_loss.png"
)


plt.savefig(
    BASELINE_LOSS_PLOT,

    dpi=200,

    bbox_inches="tight",
)


plt.close()


# ============================================================
# 17. ЗАГРУЖАЕМ ЛУЧШУЮ BASELINE
# ============================================================

best_baseline = (
    keras.models.load_model(
        BASELINE_PATH
    )
)


baseline_val_loss, baseline_val_accuracy = (
    best_baseline.evaluate(
        validation_ds,
        verbose=1,
    )
)


baseline_size_bytes = (
    BASELINE_PATH.stat().st_size
)


baseline_size_kib = (
    baseline_size_bytes / 1024
)


print()
print("=" * 60)
print("ЛУЧШАЯ BASELINE")
print("=" * 60)

print(
    f"Validation accuracy: "
    f"{baseline_val_accuracy * 100:.2f}%"
)

print(
    f"Validation loss: "
    f"{baseline_val_loss:.4f}"
)

print(
    f"Parameters: "
    f"{best_baseline.count_params()}"
)

print(
    f".keras size: "
    f"{baseline_size_kib:.2f} KiB"
)


# ============================================================
# 18. СОХРАНЯЕМ BASELINE METRICS
# ============================================================

BASELINE_METRICS_PATH = (
    METRICS_DIR /
    "baseline_validation_metrics.txt"
)


with open(
    BASELINE_METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Baseline CNN\n"
    )

    file.write(
        "==============================\n"
    )

    file.write(
        f"Classes: {class_names}\n"
    )

    file.write(
        f"Parameters: "
        f"{best_baseline.count_params()}\n"
    )

    file.write(
        f"Best epoch: "
        f"{best_epoch}\n"
    )

    file.write(
        f"Validation accuracy: "
        f"{baseline_val_accuracy:.6f}\n"
    )

    file.write(
        f"Validation loss: "
        f"{baseline_val_loss:.6f}\n"
    )

    file.write(
        f"Model size KiB: "
        f"{baseline_size_kib:.2f}\n"
    )


# ============================================================
# 19. COMPACT MODEL B
# ============================================================
#
# Baseline:
#
# 16 -> 32 -> 64 -> GAP -> Dense(32) -> Dense(3)
#
#
# Compact B:
#
# 8 -> 16 -> 32 -> GAP -> Dense(3)
#
#
# Здесь одновременно:
#
# 1. уменьшаем число convolution filters;
# 2. убираем Dense(32).
#
# Поэтому модель существенно меньше.
# ============================================================


# Снова фиксируем seed перед созданием новой модели.
keras.utils.set_random_seed(SEED)


model_b = keras.Sequential(
    [

        keras.layers.Input(
            shape=(128, 128, 3)
        ),

        keras.layers.Rescaling(
            1.0 / 255
        ),


        # ----------------------------------------------------
        # CONV BLOCK 1
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

        keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        keras.layers.GlobalAveragePooling2D(),


        keras.layers.Dense(
            units=3,
            activation="softmax",
        ),
    ]
)


# ============================================================
# 20. COMPACT B SUMMARY
# ============================================================

print()
print("=" * 60)
print("COMPACT B")
print("=" * 60)

model_b.summary()


print()

print(
    "Количество параметров Compact B:",
    model_b.count_params(),
)


# ============================================================
# 21. COMPILE COMPACT B
# ============================================================

model_b.compile(

    optimizer=keras.optimizers.Adam(),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ],
)


# ============================================================
# 22. CALLBACKS COMPACT B
# ============================================================

COMPACT_B_PATH = (
    MODELS_DIR /
    "compact_b_best.keras"
)


compact_b_checkpoint = (
    keras.callbacks.ModelCheckpoint(

        filepath=COMPACT_B_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1,
    )
)


compact_b_early_stopping = (
    keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=3,

        restore_best_weights=True,

        verbose=1,
    )
)


# ============================================================
# 23. TRAIN COMPACT B
# ============================================================

print()
print("=" * 60)
print("ОБУЧЕНИЕ COMPACT B")
print("=" * 60)


history_b = model_b.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=[
        compact_b_checkpoint,
        compact_b_early_stopping,
    ],
)


# ============================================================
# 24. МЕТРИКИ COMPACT B
# ============================================================

train_accuracy_b = (
    history_b.history["accuracy"]
)

val_accuracy_b = (
    history_b.history["val_accuracy"]
)

train_loss_b = (
    history_b.history["loss"]
)

val_loss_b = (
    history_b.history["val_loss"]
)


actual_epochs_b = len(
    history_b.history["loss"]
)


epochs_range_b = range(
    1,
    actual_epochs_b + 1,
)


best_val_accuracy_b = max(
    val_accuracy_b
)


best_epoch_b = (
    val_accuracy_b.index(
        best_val_accuracy_b
    )
    + 1
)


print()
print("COMPACT B")

print(
    "Лучший epoch:",
    best_epoch_b,
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy_b:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy_b * 100:.2f}%"
)


# ============================================================
# 25. COMPACT B ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs_range_b,
    train_accuracy_b,
    label="Train accuracy",
)


plt.plot(
    epochs_range_b,
    val_accuracy_b,
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


COMPACT_B_ACCURACY_PLOT = (
    PLOTS_DIR /
    "compact_b_accuracy.png"
)


plt.savefig(
    COMPACT_B_ACCURACY_PLOT,

    dpi=200,

    bbox_inches="tight",
)


plt.close()


# ============================================================
# 26. COMPACT B LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs_range_b,
    train_loss_b,
    label="Train loss",
)


plt.plot(
    epochs_range_b,
    val_loss_b,
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


COMPACT_B_LOSS_PLOT = (
    PLOTS_DIR /
    "compact_b_loss.png"
)


plt.savefig(
    COMPACT_B_LOSS_PLOT,

    dpi=200,

    bbox_inches="tight",
)


plt.close()


# ============================================================
# 27. ЗАГРУЖАЕМ ЛУЧШУЮ COMPACT B
# ============================================================

best_model_b = (
    keras.models.load_model(
        COMPACT_B_PATH
    )
)


compact_b_val_loss, compact_b_val_accuracy = (
    best_model_b.evaluate(
        validation_ds,
        verbose=1,
    )
)


compact_b_size_bytes = (
    COMPACT_B_PATH.stat().st_size
)


compact_b_size_kib = (
    compact_b_size_bytes / 1024
)


print()
print("=" * 60)
print("ЛУЧШАЯ COMPACT B")
print("=" * 60)


print(
    f"Validation accuracy: "
    f"{compact_b_val_accuracy * 100:.2f}%"
)

print(
    f"Validation loss: "
    f"{compact_b_val_loss:.4f}"
)

print(
    f"Parameters: "
    f"{best_model_b.count_params()}"
)

print(
    f".keras size: "
    f"{compact_b_size_kib:.2f} KiB"
)


# ============================================================
# 28. СОХРАНЯЕМ COMPACT B METRICS
# ============================================================

COMPACT_B_METRICS_PATH = (
    METRICS_DIR /
    "compact_b_validation_metrics.txt"
)


with open(
    COMPACT_B_METRICS_PATH,
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
        f"{best_model_b.count_params()}\n"
    )

    file.write(
        f"Best epoch: "
        f"{best_epoch_b}\n"
    )

    file.write(
        f"Validation accuracy: "
        f"{compact_b_val_accuracy:.6f}\n"
    )

    file.write(
        f"Validation loss: "
        f"{compact_b_val_loss:.6f}\n"
    )

    file.write(
        f"Model size KiB: "
        f"{compact_b_size_kib:.2f}\n"
    )


# ============================================================
# 29. ФИНАЛЬНОЕ СРАВНЕНИЕ
# ============================================================

print()
print("=" * 70)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 70)


print(
    f"{'Model':<15}"
    f"{'Params':>12}"
    f"{'Val accuracy':>18}"
    f"{'Val loss':>15}"
    f"{'Size KiB':>15}"
)


print("-" * 75)


print(
    f"{'Baseline':<15}"
    f"{best_baseline.count_params():>12}"
    f"{baseline_val_accuracy * 100:>17.2f}%"
    f"{baseline_val_loss:>15.4f}"
    f"{baseline_size_kib:>15.2f}"
)


print(
    f"{'Compact B':<15}"
    f"{best_model_b.count_params():>12}"
    f"{compact_b_val_accuracy * 100:>17.2f}%"
    f"{compact_b_val_loss:>15.4f}"
    f"{compact_b_size_kib:>15.2f}"
)


print()
print("Модели:")

print(
    "Baseline:",
    BASELINE_PATH,
)

print(
    "Compact B:",
    COMPACT_B_PATH,
)


print()
print(
    "ВАЖНО: test set пока не используется."
)

print(
    "Финальную модель выбираем по validation."
)