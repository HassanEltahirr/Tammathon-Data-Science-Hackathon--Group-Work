import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
from PIL import Image


DATASET_PATH = 'cat_faces'
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0001
NUM_CLASSES = 59
VALIDATION_SPLIT = 0.2
SEED = 42 

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def check_dataset(dataset_path, num_classes):
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset directory not found at '{dataset_path}'")
        print("Please ensure DATASET_PATH is set correctly.")
        return False

    breed_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    actual_num_classes = len(breed_folders)

    if actual_num_classes == 0:
        print(f"Error: No subdirectories found in '{dataset_path}'.")
        print("Please ensure the dataset directory contains folders for each breed.")
        return False

    if actual_num_classes != num_classes:
        print(f"Warning: Expected {num_classes} breed folders, but found {actual_num_classes}.")
        print(f"Please ensure NUM_CLASSES ({num_classes}) matches the number of folders in '{dataset_path}'.")

    print(f"Dataset check: Found {actual_num_classes} breed folders in '{dataset_path}'.")
    return True

print("1. Loading and Preprocessing Data...")

if not check_dataset(DATASET_PATH, NUM_CLASSES):
    exit()

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
print(f"Found {len(class_names)} classes: {class_names}")

if len(class_names) != NUM_CLASSES:
      print(f"\nError: Mismatch between NUM_CLASSES ({NUM_CLASSES}) and classes found by generator ({len(class_names)}).")
      print("Please check your DATASET_PATH and NUM_CLASSES setting.")
      exit()

print("\n2. Building Model...")

base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D(name='global_average_pooling2d')(x) # Added name for easier layer access
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("3. Compiling Model...")
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n4. Starting Training...")

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

if steps_per_epoch == 0:
    steps_per_epoch = 1
if validation_steps == 0:
    validation_steps = 1

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

print("Training Finished.")

print("\n5. Evaluating Model...")
loss, accuracy = model.evaluate(validation_generator, steps=validation_steps)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

print("\nSaving the trained model...")
model.save('cat_breed_model.keras')
print("Model saved as 'cat_breed_model.keras'")

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.close()

plot_history(history)

print("\n6. Calculating Class Similarity...")

feature_extractor_model = Model(inputs=model.input,
                                outputs=model.get_layer('global_average_pooling2d').output)

validation_generator.reset()

features = feature_extractor_model.predict(validation_generator,
                                           steps=validation_steps,
                                           verbose=1)

num_validation_samples = validation_generator.samples
if validation_generator.samples % BATCH_SIZE != 0:
     print("Warning: Validation samples not perfectly divisible by batch size. Feature/label alignment relies on generator behavior.")

validation_generator.reset()
labels = []
num_batches_to_process = validation_steps
for i in range(num_batches_to_process):
    _, batch_labels = next(validation_generator)
    labels.extend(np.argmax(batch_labels, axis=1))

num_labels_obtained = len(labels)
num_features_obtained = features.shape[0]

min_count = min(num_features_obtained, num_labels_obtained)
features = features[:min_count]
labels = labels[:min_count]

print(f"Extracted {features.shape[0]} feature vectors.")

if features.shape[0] == 0:
    print("Error: No features were extracted. Cannot calculate similarity.")
    exit()

average_features = np.zeros((NUM_CLASSES, features.shape[1]))
class_counts = np.zeros(NUM_CLASSES)

for i in range(len(labels)):
    class_idx = labels[i]
    average_features[class_idx] += features[i]
    class_counts[class_idx] += 1

valid_classes_mask = class_counts > 0

average_features[valid_classes_mask] /= class_counts[valid_classes_mask][:, np.newaxis]


if not np.all(valid_classes_mask):
    missing_classes = [class_names[i] for i, present in enumerate(valid_classes_mask) if not present]
    print(f"Warning: No validation samples found for classes: {missing_classes}. Their similarity will be based on zero vectors.")

similarity_matrix = cosine_similarity(average_features)

print("\n7. Breed Similarity Analysis Results:")

np.fill_diagonal(similarity_matrix, 0)

num_pairs_to_show = 15
indices = np.triu_indices_from(similarity_matrix, k=1)
similarity_values = similarity_matrix[indices]
sorted_indices = np.argsort(similarity_values)[::-1]

print(f"\nTop {num_pairs_to_show} Most Similar Breed Pairs (based on model features):")
for i in range(min(num_pairs_to_show, len(sorted_indices))):
    idx = sorted_indices[i]
    class1_idx = indices[0][idx]
    class2_idx = indices[1][idx]
    similarity_score = similarity_values[idx]

    class1_valid = valid_classes_mask[class1_idx]
    class2_valid = valid_classes_mask[class2_idx]

    validity_warning = ""
    if not class1_valid or not class2_valid:
        validity_warning = " (Warning: Based on zero vectors for one/both classes)"

    print(f"{i+1}. {class_names[class1_idx]} <-> {class_names[class2_idx]}: {similarity_score:.4f}{validity_warning}")

print("\nGenerating similarity heatmap...")
plt.figure(figsize=(15, 15))
plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine Similarity')
plt.title('Breed Similarity Matrix (Cosine Similarity of Average Feature Vectors)')
plt.xticks(range(NUM_CLASSES), class_names, rotation=90, fontsize=8)
plt.yticks(range(NUM_CLASSES), class_names, fontsize=8)
plt.xlabel('Breed')
plt.ylabel('Breed')
plt.tight_layout()
plt.savefig('breed_similarity_heatmap.png')
print("Similarity heatmap saved as 'breed_similarity_heatmap.png'")

print("\nScript finished.")
