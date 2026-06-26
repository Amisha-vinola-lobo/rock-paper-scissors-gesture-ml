import tensorflow as tf

# 1. Load your existing model
model = tf.keras.models.load_model('model10.h5')

# 2. Load raw datasets first (before preprocessing)
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "training data",
    image_size=(100, 100),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "training data",
    image_size=(100, 100),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# ✅ Get class names here and store them
class_names = train_ds_raw.class_names
print("Class names:", class_names)

# 3. Normalize images (convert to float and scale to 0–1)
train_ds = train_ds_raw.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds_raw.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

# 4. Compile the model again
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Fine-tune (continue training) for more epochs
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=40)

# 6. Save the updated model
model.save('model10_finetuned.h5')
print("Model fine-tuned and saved as model10_finetuned.h5")

# ❌ Do NOT call train_ds.class_names here