from pathlib import Path 

import tensorflow as tf
import keras 
import matplotlib.pyplot as plt

#1. НАСТРОЙКИ

#Размер изображений 

IMAGE_SIZE = (128, 128)

#Сеть будет есть за раз 32 картинки
BATCH_SIZE = 32

SEED = 42

# ============================================================
# 2. ПУТИ
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT/ "data"/ "prepared"

TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Не найднена папка train: {TRAIN_DIR}")

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(f"Не найдена папка validation: {VALIDATION_DIR} ")


# ============================================================
# 3. ЗАГРУЖАЕМ TRAIN
# ============================================================


train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,

    image_size = IMAGE_SIZE,

    batch_size= BATCH_SIZE,

    label_mode = "int",
    shuffle = True,

    seed = SEED,

    color_mode = "rgb",
)

# ============================================================
# 4. ЗАГРУЖАЕМ VALIDATION
# ============================================================

validation_ds = keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,

    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",

    # Validation перемешивать необязательно.
    #
    # При обучении модель validation НЕ использует
    # для обновления весов.
    shuffle=False,

    color_mode="rgb",
)

# ============================================================
# 5. СМОТРИМ, КАК KERAS НАЗНАЧИЛ КЛАССЫ
# ============================================================

class_names = train_ds.class_names 

print()
print("Классы: ")
print(class_names)


print()

for index, class_name in enumerate(class_names):
    print(f"{index} -> {class_name}")



# ============================================================
# 6. БЕРЁМ ОДИН BATCH И СМОТРИМ НА НЕГО
# ============================================================

for images, labels in train_ds.take(1):

    print()
    print("Форма: ")
    print(images.shape)

    print()
    print("Форма labels")
    print(labels.shape)

    print()
    print("Тип изображений")
    print(images.dtype)

    print()
    print("минимальное значение пикселя: ")
    print(tf.reduce_min(images).numpy())

    print("Максимальное значение пикселя:")
    print(tf.reduce_max(images).numpy())

    print()
    print("Первые labels: ")
    print(labels[:10].numpy())

# ========================================================
# 7. ПОКАЗЫВАЕМ НЕСКОЛЬКО ИЗОБРАЖЕНИЙ
# ========================================================

plt.figure(figsize=(8,8))

for i in range(9):

    plt.subplot(3, 3, i+1)

    plt.imshow(
        images[i].numpy().astype("uint8")
    )

    label_index = int(labels[i].numpy())

    label_name = class_names[label_index]

    plt.title(label_name)

    plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================================
# 8. УСКОРЯЕМ ПОДАЧУ ДАННЫХ
# ============================================================

# Пока модель вычисляет текущий batch,
# TensorFlow заранее готовит следующий.
#
# Это уменьшает простои между batch'ами.
#
# AUTOTUNE означает:
# "TensorFlow, сам выбери оптимальное количество ресурсов".


train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# 9. СОЗДАЁМ ПЕРВУЮ CNN
# ============================================================

model = keras.Sequential(
    [keras.layers.Input(
        shape = (128, 128, 3)
    ),

    #нормализация 
    keras.layers.Rescaling(
        1.0/ 255
    ),

        # --------------------------------------------------------
    # CONV BLOCK 1
    # --------------------------------------------------------

    # Создаём 16 разных фильтров размером 3x3.
    #
    # Каждый фильтр будет учиться искать какой-то
    # локальный визуальный признак:
    #
    # край,
    # направление линии,
    # текстуру,
    # переход яркости и т.д.
    #
    # padding="same" означает:
    # spatial-размер изображения пока сохраняется.
    #
    # Вход:
    # 128 x 128 x 3
    #
    # Выход:
    # 128 x 128 x 16

    keras.layers.Conv2D(
        filters = 16,
        kernel_size = (3,3),
        padding = "same",
        activation= "relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size = (2, 2)
    ),

        # --------------------------------------------------------
    # CONV BLOCK 2
    # --------------------------------------------------------

    # Теперь уже 32 фильтра.
    #
    # На более глубоких слоях сеть начинает учить
    # более сложные признаки.
    #
    # Вход:
    # 64 x 64 x 16
    #
    # Выход:
    # 64 x 64 x 32

    keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        padding = 'same',
        activation='relu'
        ),

    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),


    # --------------------------------------------------------
    # CONV BLOCK 3
    # --------------------------------------------------------

        keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
    ),

    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(
        units = 32,
        activation='relu'
    ),

    keras.layers.Dense(
        units = 3,
        activation = 'softmax'
    ),


    
              
])


model.summary()

model.compile(
    optimizer = keras.optimizers.Adam(),

    loss = keras.losses.SparseCategoricalCrossentropy(),

    metrics = ["accuracy"]
)



# ============================================================
# 12. ОБУЧЕНИЕ
# ============================================================

EPOCHS = 10

history = model.fit(

    train_ds,
    validation_data=validation_ds,

    epochs = EPOCHS

)


# ============================================================
# 13. ГРАФИК ACCURACY
# ============================================================

train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]


epochs_range = range(1, EPOCHS + 1)


plt.figure(figsize=(8, 5))

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

plt.title("Training and validation accuracy")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 14. ГРАФИК LOSS
# ============================================================

plt.figure(figsize=(8, 5))

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

plt.title("Training and validation loss")

plt.legend()

plt.tight_layout()

RESULTS_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    RESULTS_DIR / "baseline_accuracy.png",
    dpi=200,
    bbox_inches="tight",
)

plt.savefig(
    RESULTS_DIR / "baseline_loss.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()

# ============================================================
# 15. СОХРАНЕНИЕ МОДЕЛИ
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model_path = MODELS_DIR / "baseline.keras"

model.save(model_path)

print()
print("Модель сохранена:")
print(model_path)