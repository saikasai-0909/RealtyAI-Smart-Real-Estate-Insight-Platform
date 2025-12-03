# debugged_and_cleaned_realtyai.py
import os
import sys
import random
import numpy as np
import joblib as jlb
import matplotlib.pyplot as plt
import seaborn as sns

# Optional heavy imports later (lazy)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None
    layers = None
    models = None

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ----------------- Helper utilities -----------------
def check_file(path, name="file"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")

def normalize_masks(masks):
    """
    Ensure masks have shape (N, H, W) and are binary (0/1)
    """
    masks = np.array(masks)
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    return masks

# ----------------- Load datasets -----------------
ims_path = r'/home/hardik/Desktop/python_intern/archive (1)/ims.np'
mas_path = r'/home/hardik/Desktop/python_intern/archive (1)/mas.np'

check_file(ims_path, "Images file")
check_file(mas_path, "Masks file")

ims = jlb.load(ims_path)
mas = jlb.load(mas_path)

ims = np.array(ims)
mas = normalize_masks(mas)

print("Dataset loaded successfully")
print(f"Total images: {ims.shape[0]}")
print(f"Image dimensions (per image): {ims.shape[1:]}")
print(f"Mask dimensions (per mask): {mas.shape[1:]}")
print(f"Image dtype: {ims.dtype}")
print(f"Mask dtype: {mas.dtype}")

# ----------------- Batch stats (if any processed batches exist) -----------------
save_path = r'/home/hardik/Desktop/python_intern/processed_batches'
if os.path.isdir(save_path):
    batch_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.pkl') or f.endswith('.joblib')]
else:
    batch_files = []

if batch_files:
    sum_pixels = 0.0
    sum_squared_pixels = 0.0
    pixel_count = 0
    min_pixel = float('inf')
    max_pixel = float('-inf')

    for file in batch_files:
        try:
            batch = jlb.load(file)
        except Exception as e:
            print(f"Warning: failed to load batch {file}: {e}")
            continue

        # Expect batches to be (ims_batch, masks_batch) or ims_batch alone
        if isinstance(batch, tuple) or isinstance(batch, list):
            ims_batch = np.array(batch[0])
        else:
            ims_batch = np.array(batch)

        ims_batch = ims_batch.astype(np.float32)
        sum_pixels += ims_batch.sum()
        sum_squared_pixels += np.square(ims_batch).sum()
        pixel_count += ims_batch.size
        min_pixel = min(min_pixel, float(ims_batch.min()))
        max_pixel = max(max_pixel, float(ims_batch.max()))

    mean_pixel = sum_pixels / pixel_count
    std_pixel = np.sqrt(max(0.0, (sum_squared_pixels / pixel_count) - (mean_pixel ** 2)))

    print("✅ Batch Dataset Statistics:")
    print(f"Mean Pixel Value: {mean_pixel:.6f}")
    print(f"Std Dev Pixel Value: {std_pixel:.6f}")
    print(f"Min Pixel Value: {min_pixel:.6f}")
    print(f"Max Pixel Value: {max_pixel:.6f}")
else:
    print("No processed batch files found at", save_path)

# ----------------- Image shape checks -----------------
# Image resolution
if ims.ndim < 4:
    raise ValueError("Expected ims to be a 4D array (N, H, W, C). Got shape: " + str(ims.shape))

n_samples, h, w, c = ims.shape
print(f"Image height: {h}px, width: {w}px, channels: {c}")

unique_shapes = set(tuple(img.shape) for img in ims)
print(f"Unique image shapes in dataset: {unique_shapes}")

# ----------------- Mask checks -----------------
unique_mask_vals = np.unique(mas)
print("Unique mask values:", unique_mask_vals)
mask_mean = mas.mean() * 100
print(f"Average foreground pixel percentage: {mask_mean:.4f}%")

# ----------------- Pixel distributions (use a sample, not the whole dataset) -----------------
sample_idx = 0 if n_samples == 1 else min(5, n_samples-1)
sample_img = ims[sample_idx].reshape(-1, c)

plt.figure(figsize=(8, 5))
for ch in range(c):
    sns.histplot(sample_img[:, ch], bins=50, kde=True, label=f'Channel {ch}')
plt.legend()
plt.title("Pixel Intensity Distribution per Channel (sample image)")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# ----------------- Labeling: residential vs commercial (based on mask coverage) -----------------
# building_counts is number of foreground pixels per image
building_counts = np.sum(mas > 0, axis=(1, 2))  # works if mas is (N,H,W)
plt.figure(figsize=(8,4))
plt.hist(building_counts, bins=50)
plt.title("Distribution of Building Pixels per Image")
plt.xlabel("Number of Building Pixels")
plt.ylabel("Frequency")
plt.show()

# pick threshold as 90th percentile -> commercial, else residential
threshold = np.percentile(building_counts, 90)
labels = np.where(building_counts > threshold, 1, 0)
print(f"Residential (0) count: {(labels==0).sum()}")
print(f"Commercial (1) count: {(labels==1).sum()}")

# ----------------- Resize images robustly to model input size -----------------
TARGET_H, TARGET_W = 256, 256
orig_channels = c
print("Original channels:", orig_channels)

# If TensorFlow isn't available, fall back to simple numpy/resizing via PIL
def resize_images_np(ims_array, target_h, target_w):
    """Resize using PIL (works without TF)."""
    from PIL import Image
    n = ims_array.shape[0]
    resized = np.zeros((n, target_h, target_w, ims_array.shape[-1]), dtype=np.float32)
    for i in range(n):
        img = ims_array[i].astype(np.float32)
        # scale to 0-1 if 0-255
        if img.max() > 1.0:
            img = img / 255.0
        pil = Image.fromarray((np.clip(img, 0., 1.) * 255).astype(np.uint8))
        pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        # If grayscale became 2D, add channel dim
        if arr.ndim == 2:
            arr = arr[..., None]
        resized[i] = arr
    return resized

# Normalize and resize
if tf is not None:
    print("Using TensorFlow for resizing.")
    X_resized = np.zeros((n_samples, TARGET_H, TARGET_W, orig_channels), dtype=np.float32)
    for i in range(n_samples):
        img = ims[i].astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img_resized = tf.image.resize(img, (TARGET_H, TARGET_W)).numpy()
        # If TF returns shape (H,W) for single channel, expand dims
        if img_resized.ndim == 2:
            img_resized = img_resized[..., None]
        # If channel mismatch (some input images may have different channel counts), handle it
        if img_resized.shape[-1] != orig_channels:
            # try to adapt: if img_resized has 3 and orig_channels==4 -> add alpha channel zeros
            ch = img_resized.shape[-1]
            if ch < orig_channels:
                pad = np.zeros((TARGET_H, TARGET_W, orig_channels-ch), dtype=img_resized.dtype)
                img_resized = np.concatenate([img_resized, pad], axis=-1)
            else:
                img_resized = img_resized[..., :orig_channels]
        X_resized[i] = img_resized
else:
    print("TensorFlow not available — using PIL resizing.")
    X_resized = resize_images_np(ims, TARGET_H, TARGET_W)

X = X_resized
y = labels.astype(int)
print("Resized images shape:", X.shape)

# ----------------- Train / validation split (guard for single-class) -----------------
stratify_arg = y if len(np.unique(y)) > 1 else None
if stratify_arg is None:
    print("Warning: Only one class present — stratify disabled for train_test_split.")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)
print("Train set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)

# ----------------- compute class weights (guard for single-class) -----------------
unique_classes = np.unique(y_train)
if unique_classes.size > 1:
    class_weights_array = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    # compute_class_weight returns weights corresponding to `classes`, so make dict with actual labels
    class_weights = {int(cls): float(w) for cls, w in zip(unique_classes, class_weights_array)}
else:
    class_weights = {int(unique_classes[0]): 1.0}

print("Class weights:", class_weights)

# ----------------- Build simple model (requires TF) -----------------
if tf is None:
    print("TensorFlow not available — skipping model creation and training. Install TensorFlow to train.")
    sys.exit(0)

input_shape = (TARGET_H, TARGET_W, orig_channels)
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------- Train (with try/except) -----------------
try:
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=8,
        validation_data=(X_val, y_val),
        class_weight=class_weights
    )
except Exception as e:
    print("Training failed:", e)
    print("Possible reasons: GPU/CPU OOM, incorrect shapes, or invalid labels.")
    raise

# ----------------- Plot training history -----------------
plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ----------------- Evaluate -----------------
loss, acc = model.evaluate(X_val, y_val, verbose=1)
print(f"Validation Accuracy: {acc*100:.2f}%")

# ----------------- Show some predictions -----------------
indices = random.sample(range(X_val.shape[0]), min(5, X_val.shape[0]))
preds = (model.predict(X_val[indices]) > 0.5).astype(int).flatten()

plt.figure(figsize=(12,5))
for i, idx in enumerate(indices):
    plt.subplot(1, len(indices), i+1)
    # if channels >=3 show first 3 channels; else squeeze to 2D
    disp = X_val[idx]
    if disp.shape[-1] >= 3:
        plt.imshow(np.clip(disp[..., :3], 0, 1))
    else:
        plt.imshow(np.squeeze(disp), cmap='gray')
    plt.title(f"Pred: {'Com' if preds[i]==1 else 'Res'}\nTrue: {'Com' if y_val[idx]==1 else 'Res'}")
    plt.axis('off')
plt.show()

# ----------------- Save model -----------------
out_path = r'/home/hardik/Desktop/python_intern/residential_commercial_model.keras'
model.save(out_path)
print("✅ Model saved successfully at:", out_path)   
