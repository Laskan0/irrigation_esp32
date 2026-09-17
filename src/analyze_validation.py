from pathlib import Path
import csv
import math

import numpy as np
import tensorflow as tf
import keras
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

VALIDATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "prepared"
    / "validation"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline.keras"
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
    exist_ok=True
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Не найдена validation: {VALIDATION_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Не найдена модель: {MODEL_PATH}"
    )


# ============================================================
# 3. ЗАГРУЖАЕМ VALIDATION
# ============================================================

validation_ds_raw = keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",

    # Это важно:
    # порядок изображений должен оставаться фиксированным.
    shuffle=False,

    color_mode="rgb",
)

class_names = validation_ds_raw.class_names

# Пути к изображениям пригодятся,
# чтобы потом показать ошибки модели.
file_paths = validation_ds_raw.file_paths

print()
print("Классы:")
print(class_names)


# ============================================================
# 4. PREFETCH
# ============================================================

validation_ds = validation_ds_raw.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 5. ЗАГРУЖАЕМ ОБУЧЕННУЮ МОДЕЛЬ
# ============================================================

model = keras.models.load_model(
    MODEL_PATH
)

print()
print("Модель загружена:")
print(MODEL_PATH)

print()
print("Количество параметров:")
print(model.count_params())


# ============================================================
# 6. VALIDATION LOSS И ACCURACY
# ============================================================

val_loss, val_accuracy = model.evaluate(
    validation_ds,
    verbose=1
)

print()
print("Validation:")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Accuracy: {val_accuracy * 100:.2f}%")
print(f"Loss: {val_loss:.4f}")


# ============================================================
# 7. НАСТОЯЩИЕ LABELS
# ============================================================

true_labels = []

for images, labels in validation_ds:
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
    validation_ds,
    verbose=1
)

predicted_labels = np.argmax(
    predicted_probabilities,
    axis=1
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

print()
print("CLASSIFICATION REPORT")
print()
print(report_text)


report_dict = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)


# ============================================================
# 10. СОХРАНЯЕМ REPORT В TXT
# ============================================================

report_txt_path = (
    METRICS_DIR
    / "baseline_validation_report.txt"
)

with open(
    report_txt_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"Model: {MODEL_PATH.name}\n"
    )

    file.write(
        f"Parameters: {model.count_params()}\n"
    )

    file.write(
        f"Validation accuracy: {val_accuracy:.6f}\n"
    )

    file.write(
        f"Validation loss: {val_loss:.6f}\n\n"
    )

    file.write(report_text)


# ============================================================
# 11. СОХРАНЯЕМ REPORT В CSV
# ============================================================

report_csv_path = (
    METRICS_DIR
    / "baseline_validation_report.csv"
)

rows_to_save = (
    class_names
    + ["macro avg", "weighted avg"]
)

with open(
    report_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

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
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print()
print("CONFUSION MATRIX:")
print(cm)


# ============================================================
# 13. СОХРАНЯЕМ CONFUSION MATRIX В CSV
# ============================================================

cm_csv_path = (
    METRICS_DIR
    / "baseline_validation_confusion_matrix.csv"
)

with open(
    cm_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["true / predicted"]
        + class_names
    )

    for class_name, row in zip(
        class_names,
        cm
    ):

        writer.writerow(
            [class_name]
            + list(row)
        )


# ============================================================
# 14. ГРАФИК CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    ax=ax
)

ax.set_title(
    "Validation confusion matrix"
)

fig.tight_layout()

cm_plot_path = (
    PLOTS_DIR
    / "baseline_validation_confusion_matrix.png"
)

fig.savefig(
    cm_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. ГРАФИК PRECISION / RECALL / F1
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
    label="Precision"
)

ax.bar(
    x,
    recall_values,
    width,
    label="Recall"
)

ax.bar(
    x + width,
    f1_values,
    width,
    label="F1-score"
)

ax.set_xticks(x)
ax.set_xticklabels(class_names)

ax.set_ylim(0, 1)

ax.set_ylabel("Score")

ax.set_title(
    "Validation metrics by class"
)

ax.legend()

fig.tight_layout()


metrics_plot_path = (
    PLOTS_DIR
    / "baseline_validation_class_metrics.png"
)

fig.savefig(
    metrics_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. ОШИБОЧНО КЛАССИФИЦИРОВАННЫЕ ИЗОБРАЖЕНИЯ
# ============================================================

wrong_indices = np.where(
    true_labels != predicted_labels
)[0]

print()
print(
    "Количество ошибок:",
    len(wrong_indices)
)

print(
    "Всего validation изображений:",
    len(true_labels)
)


# Уверенность модели именно в её предсказании.
prediction_confidence = (
    predicted_probabilities[
        np.arange(len(predicted_labels)),
        predicted_labels
    ]
)


# Сначала покажем самые уверенные ошибки.
wrong_indices = sorted(
    wrong_indices,
    key=lambda i: prediction_confidence[i],
    reverse=True
)


NUMBER_OF_ERRORS_TO_SHOW = min(
    24,
    len(wrong_indices)
)


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
        figsize=(12, rows * 3)
    )

    axes = np.array(
        axes
    ).reshape(-1)


    for ax, index in zip(
        axes,
        selected_errors
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

        ax.imshow(image)

        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {predicted_name}\n"
            f"Conf: {confidence:.2f}"
        )

        ax.axis("off")


    # Если ошибок меньше, чем клеток графика.
    for ax in axes[
        NUMBER_OF_ERRORS_TO_SHOW:
    ]:

        ax.axis("off")


    fig.suptitle(
        "Misclassified validation images"
    )

    fig.tight_layout()


    errors_plot_path = (
        PLOTS_DIR
        / "baseline_validation_errors.png"
    )

    fig.savefig(
        errors_plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 17. ГОТОВО
# ============================================================

print()
print("Результаты сохранены:")

print(report_txt_path)
print(report_csv_path)
print(cm_csv_path)
print(cm_plot_path)
print(metrics_plot_path)

if NUMBER_OF_ERRORS_TO_SHOW > 0:
    print(errors_plot_path)