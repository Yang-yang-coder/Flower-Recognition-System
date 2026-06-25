import tensorflow as tf
from tensorflow.keras import layers, Sequential

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# 1. 数据增强
# ======================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# ======================
# 2. 加载数据
# ======================
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("类别：", class_names)

# ======================
# 3. 性能优化（关键）
# ======================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ======================
# 4. 迁移学习模型
# ======================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # 第一阶段冻结

# ======================
# 5. 构建模型
# ======================
model = Sequential([
    data_augmentation,

    layers.Rescaling(1./255),  # ⭐ 标准化

    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# 6. 第一阶段训练
# ======================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ======================
# 7. 微调（解冻）
# ======================
base_model.trainable = True

# 只训练后30层（关键优化）
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# 8. 第二阶段训练
# ======================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# ======================
# 9. 保存模型
# ======================
model.save("plant_model.h5")

print("模型训练完成！")