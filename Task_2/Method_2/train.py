import tensorflow as tf
import numpy as np
import os
import random
import math
import glob
import cv2

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 15
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
INITIAL_LEARNING_RATE = 0.0001
FINE_TUNE_LEARNING_RATE = INITIAL_LEARNING_RATE / 10

CHECKPOINT_DIR = './checkpoints'
DATA_DIR = '.'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
FACE_CASCADE_PATH = './haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = './haarcascade_eye.xml'
GLASSES_CASCADE_PATH = './haarcascade_eye_tree_eyeglasses.xml'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if not os.path.exists(FACE_CASCADE_PATH): raise FileNotFoundError(f"Face cascade not found: {FACE_CASCADE_PATH}")
if not os.path.exists(EYE_CASCADE_PATH): raise FileNotFoundError(f"Eye cascade not found: {EYE_CASCADE_PATH}")
if not os.path.exists(GLASSES_CASCADE_PATH): raise FileNotFoundError(f"Glasses cascade not found: {GLASSES_CASCADE_PATH}")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
glasses_cascade = cv2.CascadeClassifier(GLASSES_CASCADE_PATH)

N_FACE_FEATURES = 8

def detect_face_features_cv(image_tensor):
    image_rgb = image_tensor.numpy()

    img_h, img_w = image_rgb.shape[:2]
    if img_h == 0 or img_w == 0:
        print("Warning: Received empty image tensor.")
        return np.zeros(N_FACE_FEATURES, dtype=np.float32)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    try:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(0.1*img_w), int(0.1*img_h)))
    except cv2.error as e:
        print(f"OpenCV error during face detection: {e}. Skipping image.")
        return np.zeros(N_FACE_FEATURES, dtype=np.float32)

    default_features = np.zeros(N_FACE_FEATURES, dtype=np.float32)
    if len(faces) == 0:
        return default_features

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]

    if w <= 0 or h <= 0:
        print(f"Warning: Detected face with non-positive dimensions (w={w}, h={h}). Skipping ROI processing.")
        face_center_x = (x + w / 2) / img_w if img_w > 0 else 0
        face_center_y = (y + h / 2) / img_h if img_h > 0 else 0
        face_width = w / img_w if img_w > 0 else 0
        face_height = h / img_h if img_h > 0 else 0
        features = np.array([face_center_x, face_center_y, face_width, face_height, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return features

    face_center_x = (x + w / 2) / img_w if img_w > 0 else 0
    face_center_y = (y + h / 2) / img_h if img_h > 0 else 0
    face_width = w / img_w if img_w > 0 else 0
    face_height = h / img_h if img_h > 0 else 0

    y_end, x_end = min(y + h, img_h), min(x + w, img_w)
    roi_gray = gray[y:y_end, x:x_end]

    avg_eye_y_rel = 0.0
    eye_dist_rel = 0.0
    glasses_detected_flag = 0.0

    if roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
        try:
            minSize_eye_w = max(1, int(0.1 * w))
            minSize_eye_h = max(1, int(0.1 * h))
            minSize_glasses_w = max(1, int(0.2 * w))
            minSize_glasses_h = max(1, int(0.1 * h))

            if roi_gray.shape[1] >= minSize_eye_w and roi_gray.shape[0] >= minSize_eye_h:
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(minSize_eye_w, minSize_eye_h))
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda e: (e[1], e[0]))[:2]
                    avg_eye_y = np.mean([(ey + eh / 2) for ex, ey, ew, eh in eyes])
                    avg_eye_y_rel = avg_eye_y / h
                    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
                    eye_center1_x, eye_center1_y = ex1 + ew1 / 2, ey1 + eh1 / 2
                    eye_center2_x, eye_center2_y = ex2 + ew2 / 2, ey2 + eh2 / 2
                    eye_dist = math.sqrt((eye_center1_x - eye_center2_x)**2 + (eye_center1_y - eye_center2_y)**2)
                    eye_dist_rel = eye_dist / w

            if roi_gray.shape[1] >= minSize_glasses_w and roi_gray.shape[0] >= minSize_glasses_h:
                glasses = glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(minSize_glasses_w, minSize_glasses_h))
                if len(glasses) > 0:
                    glasses_detected_flag = 1.0

        except cv2.error as e:
            print(f"OpenCV error during eye/glasses detection in ROI: {e}. Using default eye/glasses features.")

    features = np.array([
        face_center_x, face_center_y, face_width, face_height,
        avg_eye_y_rel, eye_dist_rel,
        1.0,
        glasses_detected_flag
    ], dtype=np.float32)

    return features

@tf.function
def tf_detect_face_features(image_tensor_uint8):
    features = tf.py_function(
        func=detect_face_features_cv,
        inp=[image_tensor_uint8],
        Tout=tf.float32
    )
    features.set_shape((N_FACE_FEATURES,))
    return features

@tf.function
def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    img_uint8 = tf.cast(img, tf.uint8)

    face_features = tf_detect_face_features(img_uint8)

    img_resized = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.AREA)

    img_float = tf.cast(img_resized, tf.float32)

    img_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(img_float)

    return img_normalized, face_features

def configure_dataset(dataset, shuffle=False, augment=False):
    if shuffle:
        cardinality = tf.data.experimental.cardinality(dataset)
        buffer_size = min(cardinality.numpy() if cardinality > 0 else 10000, 10000)
        print(f"Shuffle buffer size: {buffer_size}")
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def create_tf_dataset(data_dir, shuffle=False, augment=False):
    image_paths = []
    labels = []
    class_names = ['fail', 'pass']
    print(f"Searching for images in: {data_dir}")

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue

        print(f"Searching in class directory: {class_dir}")
        files_found_in_class = 0
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif'):
            found_files = glob.glob(os.path.join(class_dir, ext))
            image_paths.extend(found_files)
            labels.extend([class_index] * len(found_files))
            files_found_in_class += len(found_files)
        print(f"Found {files_found_in_class} images for class '{class_name}'")

    if not image_paths:
        raise ValueError(f"No images found in {data_dir}. Check paths and extensions ('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif').")

    print(f"Total images found in {data_dir}: {len(image_paths)}")

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

    processed_ds = image_label_ds.map(
        lambda path, label: (load_and_preprocess_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    final_ds = processed_ds.map(
        lambda processed_data, label: ((processed_data[0], processed_data[1]), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    final_ds = configure_dataset(final_ds, shuffle=shuffle, augment=augment)

    return final_ds, labels

print("Creating Training Dataset...")
try:
    train_ds, train_labels_list = create_tf_dataset(TRAIN_DIR, shuffle=True, augment=False)
except ValueError as e:
    print(f"Error creating training dataset: {e}")
    exit()

print("\nCreating Validation Dataset...")
try:
    val_ds, val_labels_list = create_tf_dataset(VAL_DIR, shuffle=False, augment=False)
    validation_data_param = val_ds if tf.data.experimental.cardinality(val_ds).numpy() > 0 else None
except ValueError as e:
    print(f"Warning: Error creating validation dataset: {e}. Proceeding without validation.")
    val_ds = None
    validation_data_param = None
    val_labels_list = []

unique_classes = np.unique(train_labels_list)
class_weight_dict = None
if len(unique_classes) > 1 and len(train_labels_list) > 0:
    try:
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels_list)
        class_weight_dict = dict(enumerate(class_weights))
        print(f"\nCalculated Class Weights: {class_weight_dict}")
    except ValueError as e:
        print(f"\nWarning: Could not compute class weights ({e}). Training without class weights.")
elif len(train_labels_list) == 0:
    print("\nError: No training labels found. Cannot compute class weights.")
else:
    print(f"\nWarning: Only one class ({unique_classes[0]}) found in training data. Class weights not applicable.")

def create_passport_classifier(img_size, n_face_features, num_classes=1):
    img_input = Input(shape=(*img_size, 3), name='image_input')
    face_input = Input(shape=(n_face_features,), name='face_feature_input')

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False

    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5, seed=SEED)(x)
    image_features = Dense(64, activation='relu', name='image_features')(x)

    y = Dense(32, activation='relu')(face_input)
    y = Dropout(0.3, seed=SEED)(y)
    face_processed_features = Dense(16, activation='relu', name='face_processed_features')(y)

    combined = Concatenate()([image_features, face_processed_features])

    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5, seed=SEED)(z)

    if num_classes == 1:
        activation = 'sigmoid'
        loss = BinaryCrossentropy()
    else:
        activation = 'softmax'
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    output = Dense(num_classes, activation=activation, name='output')(z)

    model = Model(inputs=[img_input, face_input], outputs=output)

    return model, loss, base_model

print("\nBuilding the model...")
model, loss_function, base_model_instance = create_passport_classifier(IMG_SIZE, N_FACE_FEATURES, num_classes=1)

metrics = [
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    Precision(thresholds=0.5, class_id=1, name='precision_pass'),
    Recall(thresholds=0.5, class_id=1, name='recall_pass'),
    Recall(thresholds=0.5, class_id=0, name='recall_fail')
]

model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
              loss=loss_function,
              metrics=metrics)

print("--- Model Summary (Before Fine-tuning) ---")
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
                              min_lr=max(INITIAL_LEARNING_RATE / 100, FINE_TUNE_LEARNING_RATE / 10),
                              verbose=1)

checkpoint_filepath_best = os.path.join(CHECKPOINT_DIR, 'best_model_epoch_{epoch:02d}_valloss_{val_loss:.3f}.keras')
model_checkpoint_best = ModelCheckpoint(
    filepath=checkpoint_filepath_best,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
# callbacks_list = [reduce_lr, model_checkpoint_best, early_stopping]
callbacks_list = [reduce_lr, model_checkpoint_best]

print(f"\n--- Starting Initial Training ({INITIAL_EPOCHS} Epochs) ---")
history = None
if train_ds:
    history = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_data_param,
        class_weight=class_weight_dict,
        callbacks=callbacks_list
    )
    print("\n--- Initial Training Finished ---")
else:
    print("\n--- Skipping Initial Training: Training dataset not available. ---")

history_fine = None
if history:
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'best_model_epoch_*.keras'))
    best_checkpoint_path = None
    if checkpoint_files:
         best_checkpoint_path = max(checkpoint_files, key=os.path.getctime)

    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        print(f"Loading best weights from checkpoint: {best_checkpoint_path}")
        try:
            model = tf.keras.models.load_model(best_checkpoint_path)
            print("Successfully loaded model from .keras file.")
            base_model_instance = model.get_layer(index=1)
        except Exception as e:
            print(f"Could not load full model from {best_checkpoint_path}: {e}. Attempting to load weights only.")
            try:
                 model.load_weights(best_checkpoint_path)
                 print("Successfully loaded weights into existing model structure.")
            except Exception as e_weights:
                 print(f"Could not load weights either from {best_checkpoint_path}: {e_weights}")
                 print("Proceeding with weights from the end of initial training.")
    else:
        print("No best checkpoint file (.keras) found in the specified format. Continuing with current model weights.")


    print(f"\n--- Starting Fine-Tuning ({FINE_TUNE_EPOCHS} Epochs) ---")
    base_model_instance.trainable = True
    print(f"Base model '{base_model_instance.name}' unfrozen.")

    model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
                  loss=loss_function,
                  metrics=metrics)

    print("--- Model Summary (During Fine-tuning) ---")
    model.summary()

    history_fine = model.fit(
        train_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=validation_data_param,
        class_weight=class_weight_dict,
        callbacks=callbacks_list
    )
    print("\n--- Fine-Tuning Finished ---")
else:
    print("\n--- Skipping Fine-Tuning: Initial training did not occur. ---")

if validation_data_param:
    print("\nEvaluating final model on validation data...")
    try:
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'best_model_epoch_*.keras'))
        best_checkpoint_path_final = None
        if checkpoint_files:
             best_checkpoint_path_final = max(checkpoint_files, key=os.path.getctime)

        if best_checkpoint_path_final and os.path.exists(best_checkpoint_path_final):
            print(f"Loading best weights from final best checkpoint: {best_checkpoint_path_final}")
            try:
                model = tf.keras.models.load_model(best_checkpoint_path_final)
                print("Successfully loaded best model for final evaluation.")
            except Exception as e:
                print(f"Could not load best model from {best_checkpoint_path_final}: {e}. Evaluating with final weights.")
        else:
            print("No best checkpoint file (.keras) found. Evaluating with final weights from training.")

        results = model.evaluate(val_ds, verbose=1)
        print("\n--- Final Validation Results (using best saved model) ---")
        print(f" Loss: {results[0]:.4f}")
        results_dict = dict(zip(model.metrics_names, results))
        for name, value in results_dict.items():
            print(f" {name.replace('_', ' ').capitalize()}: {value:.4f}")

        precision = results_dict.get('precision')
        recall = results_dict.get('recall')
        if precision is not None and recall is not None:
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)
            print(f" F1 Score (Overall): {f1_score:.4f}")
        else:
            print(" F1 Score (Overall): Could not calculate (Precision or Recall missing).")

    except tf.errors.InvalidArgumentError as e:
        print(f"\nError during evaluation (InvalidArgumentError): {e}")
        print("This might happen if the validation data is empty or has unexpected dimensions.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during evaluation: {e}")
else:
    print("\nSkipping final evaluation as validation data was not available or failed to load.")

def plot_history(history, fine_tune_history=None, initial_epochs=0):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    recall_pass = history.history.get('recall_pass', [])
    val_recall_pass = history.history.get('val_recall_pass', [])

    if fine_tune_history:
        acc += fine_tune_history.history.get('accuracy', [])
        val_acc += fine_tune_history.history.get('val_accuracy', [])
        loss += fine_tune_history.history.get('loss', [])
        val_loss += fine_tune_history.history.get('val_loss', [])
        recall_pass += fine_tune_history.history.get('recall_pass', [])
        val_recall_pass += fine_tune_history.history.get('val_recall_pass', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    if acc or val_acc:
        if acc: plt.plot(epochs_range, acc, label='Training Accuracy')
        if val_acc: plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        if fine_tune_history and initial_epochs > 0:
             plt.axvline(initial_epochs -1, linestyle='--', color='r', label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Accuracy data unavailable', ha='center', va='center')

    plt.subplot(2, 2, 2)
    if loss or val_loss:
        if loss: plt.plot(epochs_range, loss, label='Training Loss')
        if val_loss: plt.plot(epochs_range, val_loss, label='Validation Loss')
        if fine_tune_history and initial_epochs > 0:
            plt.axvline(initial_epochs-1, linestyle='--', color='r', label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Loss data unavailable', ha='center', va='center')

    plt.subplot(2, 2, 3)
    if recall_pass or val_recall_pass:
        if recall_pass: plt.plot(epochs_range, recall_pass, label='Training Recall (Pass)')
        if val_recall_pass: plt.plot(epochs_range, val_recall_pass, label='Validation Recall (Pass)')
        if fine_tune_history and initial_epochs > 0:
            plt.axvline(initial_epochs -1, linestyle='--', color='r', label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Recall (Pass Class)')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Recall (Pass) data unavailable', ha='center', va='center')

    # plt.subplot(2, 2, 4)
    # ... plotting code for another metric ...

    plt.tight_layout()
    plt.show()

if 'history' in locals() and history is not None and history.history:
    fine_history_obj = history_fine if 'history_fine' in locals() and history_fine is not None else None
    plot_history(history, fine_tune_history=fine_history_obj, initial_epochs=INITIAL_EPOCHS)
else:
    print("\nSkipping plotting as training did not complete successfully or history object is missing/empty.")

print("\nScript finished.")
