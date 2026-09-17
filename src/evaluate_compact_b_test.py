from pathlib import Path
import csv
import math

import numpy as np
import tensorflow as tf
import keras

# Чтобы matplotlib не падал на macOS
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


# ============================================================
# 2. ПУТИ
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "prepared"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "compact_b_best.keras"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
)

PLOTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "plots"
)


METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Не найдена test-папка: {TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Не найдена модель: {MODEL_PATH}"
    )


# ============================================================
# 3. ЗАГРУЖАЕМ TEST
# ============================================================
#
# ВАЖНО:
#
# shuffle=False нужен обязательно.
#
# Нам необходимо сохранить соответствие:
#
# изображение -> настоящий label -> prediction
#
# чтобы потом корректно построить confusion matrix
# и показать ошибочно классифицированные изображения.
# ============================================================

test_ds_raw = keras.utils.image_dataset_from_directory(
    TEST_DIR,

    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=False,

    color_mode="rgb",
)


class_names = test_ds_raw.class_names

file_paths = test_ds_raw.file_paths


print()
print("Классы:")

for index, class_name in enumerate(class_names):
    print(
        f"{index} -> {class_name}"
    )


print()
print(
    "Количество test-изображений:",
    len(file_paths),
)


# ============================================================
# 4. PREFETCH
# ============================================================

test_ds = test_ds_raw.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 5. ЗАГРУЖАЕМ FINAL COMPACT B
# ============================================================

model = keras.models.load_model(
    MODEL_PATH
)


print()
print("=" * 60)
print("FINAL COMPACT B")
print("=" * 60)

print()

print(
    "Модель:",
    MODEL_PATH,
)

print(
    "Количество параметров:",
    model.count_params(),
)


# ============================================================
# 6. TEST LOSS И ACCURACY
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1,
)


print()
print("=" * 60)
print("TEST RESULT")
print("=" * 60)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test loss: "
    f"{test_loss:.4f}"
)


# ============================================================
# 7. НАСТОЯЩИЕ LABELS
# ============================================================

true_labels = []


for images, labels in test_ds:

    true_labels.extend(
        labels.numpy()
    )


true_labels = np.array(
    true_labels
)


# ============================================================
# 8. ПРЕДСКАЗАНИЯ МОДЕЛИ
# ============================================================

predicted_probabilities = model.predict(
    test_ds,
    verbose=1,
)


predicted_labels = np.argmax(
    predicted_probabilities,
    axis=1,
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(
    true_labels,
    predicted_labels,

    target_names=class_names,

    digits=4,

    zero_division=0,
)


report_dict = classification_report(
    true_labels,
    predicted_labels,

    target_names=class_names,

    output_dict=True,

    zero_division=0,
)


print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print()

print(
    report_text
)


# ============================================================
# 10. MACRO F1
# ============================================================

macro_f1 = (
    report_dict["macro avg"]["f1-score"]
)


print(
    f"Macro-F1: {macro_f1:.4f}"
)

print(
    f"Macro-F1: {macro_f1 * 100:.2f}%"
)


# ============================================================
# 11. СОХРАНЯЕМ REPORT В TXT
# ============================================================

REPORT_TXT_PATH = (
    METRICS_DIR
    / "compact_b_test_report.txt"
)


with open(
    REPORT_TXT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "FINAL TEST — COMPACT B\n"
    )

    file.write(
        "========================================\n"
    )

    file.write(
        f"Model: {MODEL_PATH.name}\n"
    )

    file.write(
        f"Parameters: {model.count_params()}\n"
    )

    file.write(
        f"Classes: {class_names}\n"
    )

    file.write(
        f"Test images: {len(true_labels)}\n"
    )

    file.write(
        f"Test accuracy: {test_accuracy:.6f}\n"
    )

    file.write(
        f"Test loss: {test_loss:.6f}\n"
    )

    file.write(
        f"Macro-F1: {macro_f1:.6f}\n\n"
    )

    file.write(
        report_text
    )


# ============================================================
# 12. СОХРАНЯЕМ REPORT В CSV
# ============================================================

REPORT_CSV_PATH = (
    METRICS_DIR
    / "compact_b_test_report.csv"
)


rows_to_save = (
    class_names
    + [
        "macro avg",
        "weighted avg",
    ]
)


with open(
    REPORT_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([
        "class",
        "precision",
        "recall",
        "f1_score",
        "support",
    ])


    for name in rows_to_save:

        metrics = report_dict[name]

        writer.writerow([
            name,
            metrics["precision"],
            metrics["recall"],
            metrics["f1-score"],
            metrics["support"],
        ])


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
)


print()
print("=" * 60)
print("TEST CONFUSION MATRIX")
print("=" * 60)

print(
    cm
)


# ============================================================
# 14. СОХРАНЯЕМ CONFUSION MATRIX В CSV
# ============================================================

CM_CSV_PATH = (
    METRICS_DIR
    / "compact_b_test_confusion_matrix.csv"
)


with open(
    CM_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        ["true / predicted"]
        + class_names
    )


    for class_name, row in zip(
        class_names,
        cm,
    ):

        writer.writerow(
            [class_name]
            + list(row)
        )


# ============================================================
# 15. ГРАФИК CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)


display.plot(
    ax=ax,
)


ax.set_title(
    "Compact B — Test confusion matrix"
)


fig.tight_layout()


CM_PLOT_PATH = (
    PLOTS_DIR
    / "compact_b_test_confusion_matrix.png"
)


fig.savefig(
    CM_PLOT_PATH,

    dpi=200,

    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# 16. PRECISION / RECALL / F1 ПО КЛАССАМ
# ============================================================

precision_values = [
    report_dict[name]["precision"]
    for name in class_names
]


recall_values = [
    report_dict[name]["recall"]
    for name in class_names
]


f1_values = [
    report_dict[name]["f1-score"]
    for name in class_names
]


x = np.arange(
    len(class_names)
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    x - width,
    precision_values,
    width,
    label="Precision",
)


ax.bar(
    x,
    recall_values,
    width,
    label="Recall",
)


ax.bar(
    x + width,
    f1_values,
    width,
    label="F1-score",
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    class_names
)

ax.set_ylim(
    0,
    1,
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Compact B — Test metrics by class"
)

ax.legend()


fig.tight_layout()


METRICS_PLOT_PATH = (
    PLOTS_DIR
    / "compact_b_test_class_metrics.png"
)


fig.savefig(
    METRICS_PLOT_PATH,

    dpi=200,

    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# 17. ОШИБОЧНО КЛАССИФИЦИРОВАННЫЕ TEST-ИЗОБРАЖЕНИЯ
# ============================================================

wrong_indices = np.where(
    true_labels
    != predicted_labels
)[0]


print()
print(
    "Количество ошибок:",
    len(wrong_indices),
)

print(
    "Всего test-изображений:",
    len(true_labels),
)


# ============================================================
# 18. CONFIDENCE
# ============================================================
#
# Для каждого изображения берём вероятность класса,
# который модель выбрала.
# ============================================================

prediction_confidence = (
    predicted_probabilities[
        np.arange(
            len(predicted_labels)
        ),
        predicted_labels,
    ]
)


# Сначала смотрим ошибки,
# в которых модель была наиболее уверена.
wrong_indices = sorted(
    wrong_indices,

    key=lambda i:
        prediction_confidence[i],

    reverse=True,
)


NUMBER_OF_ERRORS_TO_SHOW = min(
    24,
    len(wrong_indices),
)


# ============================================================
# 19. ГРАФИК ОШИБОК
# ============================================================

ERRORS_PLOT_PATH = None


if NUMBER_OF_ERRORS_TO_SHOW > 0:

    selected_errors = wrong_indices[
        :NUMBER_OF_ERRORS_TO_SHOW
    ]


    columns = 4

    rows = math.ceil(
        NUMBER_OF_ERRORS_TO_SHOW
        / columns
    )


    fig, axes = plt.subplots(
        rows,
        columns,

        figsize=(
            12,
            rows * 3,
        ),
    )


    axes = np.array(
        axes
    ).reshape(-1)


    for ax, index in zip(
        axes,
        selected_errors,
    ):

        image = keras.utils.load_img(
            file_paths[index]
        )


        true_name = class_names[
            true_labels[index]
        ]


        predicted_name = class_names[
            predicted_labels[index]
        ]


        confidence = (
            prediction_confidence[index]
        )


        ax.imshow(
            image
        )


        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {predicted_name}\n"
            f"Conf: {confidence:.2f}"
        )


        ax.axis(
            "off"
        )


    # Выключаем незаполненные subplot.
    for ax in axes[
        NUMBER_OF_ERRORS_TO_SHOW:
    ]:

        ax.axis(
            "off"
        )


    fig.suptitle(
        "Compact B — Misclassified test images"
    )


    fig.tight_layout()


    ERRORS_PLOT_PATH = (
        PLOTS_DIR
        / "compact_b_test_errors.png"
    )


    fig.savefig(
        ERRORS_PLOT_PATH,

        dpi=200,

        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# 20. ФИНАЛЬНЫЙ РЕЗУЛЬТАТ
# ============================================================

print()
print("=" * 60)
print("FINAL TEST ЗАВЕРШЁН")
print("=" * 60)

print()

print(
    f"Model: "
    f"{MODEL_PATH.name}"
)

print(
    f"Parameters: "
    f"{model.count_params()}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro-F1: "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"Ошибок: "
    f"{len(wrong_indices)} / {len(true_labels)}"
)


print()
print("Результаты сохранены:")

print(
    REPORT_TXT_PATH
)

print(
    REPORT_CSV_PATH
)

print(
    CM_CSV_PATH
)

print(
    CM_PLOT_PATH
)

print(
    METRICS_PLOT_PATH
)


if ERRORS_PLOT_PATH is not None:

    print(
        ERRORS_PLOT_PATH
    )