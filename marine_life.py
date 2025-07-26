import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, LeakyReLU, ELU
from tensorflow.keras.applications import MobileNetV2, ResNet101
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
import os
import random
from PIL import Image
import logging
import math
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
tf.get_logger().setLevel('ERROR')

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_random_seed()

IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 30
NUM_CLASSES = 22
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', min_lr=0.0000001)

checkpoint_filepath = 'marine_life_resnet101.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint_callback]

base_path = 'marine_life_split'

def create_dataframe_from_folder(base_dir):
    filepaths = []
    labels = []
    data_sets = []

    for split_name in ['train', 'test', 'val']:
        split_path = os.path.join(base_dir, split_name)
        if not os.path.isdir(split_path):
            logging.warning(f"Split folder '{split_path}' not found. Skipping this split.")
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        full_path = os.path.join(class_path, file)
                        relative_path = os.path.relpath(full_path, base_dir)

                        filepaths.append(relative_path)
                        labels.append(class_name)
                        data_sets.append(split_name)

    new_df = pd.DataFrame({
        'filepaths': filepaths,
        'labels': labels,
        'image_path': [os.path.join(base_dir, fp) for fp in filepaths],
        'data set': data_sets
    })

    return new_df

logging.info(f"Scanning '{base_path}' to create initial DataFrame from folder structure...")
df = create_dataframe_from_folder(base_path)

if df.empty:
    raise ValueError(f"No image files found in '{base_path}'. Please check your base_path and folder structure.")

train_df_original = df[df['data set'] == 'train'].copy()
test_df_original = df[df['data set'] == 'test'].copy()
validation_df_original = df[df['data set'] == 'val'].copy()

train_df_original = train_df_original.rename(columns={'labels': 'label'})
test_df_original = test_df_original.rename(columns={'labels': 'label'})
validation_df_original = validation_df_original.rename(columns={'labels': 'label'})

logging.info("\n--- Initial Data Distribution (Text Summary) ---")

logging.info("\nTRAIN Split:")
train_class_distribution = train_df_original['label'].value_counts().sort_index()
logging.info(f"Number of classes: {len(train_class_distribution)}")
logging.info(f"Total images: {len(train_df_original)}")
logging.info("Class-wise image counts:")
for class_name, count in train_class_distribution.items():
    logging.info(f"  {class_name}: {count} images")
logging.info(f"Minimum images per class: {train_class_distribution.min()} images")
logging.info(f"Maximum images per class: {train_class_distribution.max()} images")

logging.info("\nVAL Split:")
val_class_distribution = validation_df_original['label'].value_counts().sort_index()
logging.info(f"Number of classes: {len(val_class_distribution)}")
logging.info(f"Total images: {len(validation_df_original)}")
logging.info("Class-wise image counts:")
for class_name, count in val_class_distribution.items():
    logging.info(f"  {class_name}: {count} images")
logging.info(f"Minimum images per class: {val_class_distribution.min()} images")
logging.info(f"Maximum images per class: {val_class_distribution.max()} images")

logging.info("\nTEST Split:")
test_class_distribution = test_df_original['label'].value_counts().sort_index()
logging.info(f"Number of classes: {len(test_class_distribution)}")
logging.info(f"Total images: {len(test_df_original)}")
logging.info("Class-wise image counts:")
for class_name, count in test_class_distribution.items():
    logging.info(f"  {class_name}: {count} images")
logging.info(f"Minimum images per class: {test_class_distribution.min()} images")
logging.info(f"Maximum images per class: {test_class_distribution.max()} images")

temp_datagen = ImageDataGenerator()
temp_generator = temp_datagen.flow_from_dataframe(
    dataframe=df.rename(columns={'labels': 'label'}),
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

logging.info(f"\nDetected NUM_CLASSES: {NUM_CLASSES}")

logging.info("\n--- Class Serialization (Class Name to Index Mapping) ---")
sorted_class_indices = sorted(temp_generator.class_indices.items(), key=lambda item: item[1])
for class_name, index in sorted_class_indices:
    logging.info(f"  {class_name}: {index}")

overall_class_distribution = df['labels'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=overall_class_distribution.index, y=overall_class_distribution.values, palette='cubehelix')
plt.title('Overall Distribution of Classes Across All Data (Train, Test, Val Combined)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

logging.info("\nOverall Class Distribution (from plot data):")
logging.info(overall_class_distribution)

train_class_distribution = train_df_original['label'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=train_class_distribution.index, y=train_class_distribution.values, palette='viridis')
plt.title('Distribution of Classes in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()
logging.info("\nTraining Set Class Distribution (from plot data):")
logging.info(train_class_distribution)

val_class_distribution = validation_df_original['label'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=val_class_distribution.index, y=val_class_distribution.values, palette='magma')
plt.title('Distribution of Classes in Validation Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()
logging.info("\nValidation Set Class Distribution (from plot data):")
logging.info(val_class_distribution)

test_class_distribution = test_df_original['label'].value_counts()
plt.figure(figsize=(14, 7))
sns.barplot(x=test_class_distribution.index, y=test_class_distribution.values, palette='cividis')
plt.title('Distribution of Classes in Test Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()
logging.info("\nTest Set Class Distribution (from plot data):")
logging.info(test_class_distribution)

data_volume = pd.DataFrame({
    'Dataset': ['Train', 'Test', 'Validation'],
    'Count': [len(train_df_original), len(test_df_original), len(validation_df_original)]
})
plt.figure(figsize=(8, 5))
sns.barplot(x='Dataset', y='Count', data=data_volume, palette='coolwarm')
plt.title('Volume of Train, Test, and Validation Datasets')
plt.xlabel('Dataset Type')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.show()

dataset_type_distribution = df['data set'].value_counts()
plt.figure(figsize=(7, 5))
plt.pie(dataset_type_distribution, labels=dataset_type_distribution.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution of Samples by Dataset Type')
plt.axis('equal')
plt.show()

unique_classes_count = pd.DataFrame({
    'Dataset': ['Train', 'Test', 'Validation'],
    'Unique Classes': [train_df_original['label'].nunique(), test_df_original['label'].nunique(), validation_df_original['label'].nunique()]
})
plt.figure(figsize=(8, 5))
sns.barplot(x='Dataset', y='Unique Classes', data=unique_classes_count, palette='rocket')
plt.title('Number of Unique Classes per Dataset Split')
plt.xlabel('Dataset Type')
plt.ylabel('Count of Unique Classes')
plt.tight_layout()
plt.show()

def display_sample_images(dataframe, num_classes=5, images_per_class=3):
    unique_classes = dataframe['label'].unique()
    if len(unique_classes) < num_classes:
        num_classes = len(unique_classes)

    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)

    plt.figure(figsize=(images_per_class * 3, num_classes * 3))
    plt.suptitle('Sample Images from Different Classes', fontsize=18, y=1.02)

    for i, class_name in enumerate(selected_classes):
        class_images = dataframe[dataframe['label'] == class_name]['image_path'].sample(min(images_per_class, len(dataframe[dataframe['label'] == class_name])), random_state=42)

        for j, image_path in enumerate(class_images):
            ax = plt.subplot(num_classes, images_per_class, i * images_per_class + j + 1)
            try:
                img = Image.open(image_path)
                ax.imshow(img)
                ax.set_title(f"{class_name}", fontsize=10)
                ax.axis('off')
            except Exception as e:
                ax.set_title(f"Error loading {class_name}", fontsize=10)
                ax.axis('off')
                logging.error(f"Could not load image {image_path}: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

logging.info("\n--- Displaying Sample Images ---")
display_sample_images(train_df_original, num_classes=5, images_per_class=3)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)

try:
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df_original,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=validation_df_original,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df_original,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
except Exception as e:
    logging.error(f"Error loading data generators: {e}")
    raise

logging.info(f"Class indices from train_generator: {train_generator.class_indices}")

def create_model(base_model_class, input_shape, num_classes):
    model = Sequential()
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    model.add(base_model)

    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(256, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    return model

logging.info("--- Starting Feature Extraction Training for ResNet-101 Model (Base Frozen) ---")

resnet101 = create_model(ResNet101, input_shape, NUM_CLASSES)

total_train_steps = math.ceil(train_generator.samples / BATCH_SIZE) * EPOCHS
lr_schedule = CosineDecay(
    initial_learning_rate=0.0001,
    decay_steps=total_train_steps,
    alpha=0.01
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

resnet101.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

resnet101.summary(print_fn=logging.info)

print(len(resnet101.layers))

resnet101_history = resnet101.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / BATCH_SIZE),
    callbacks=callbacks
)

resnet101_loss, resnet101_acc = resnet101.evaluate(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))

logging.info(f'ResNet-101 Test Accuracy: {resnet101_acc:.4f}')
logging.info(f'ResNet-101 Test Loss: {resnet101_loss:.4f}')

def plot_augmented_images(generator, num_images=9):
    images, labels = next(generator)

    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(images))):
        plt.subplot(3, 3, i + 1)
        display_image = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
        plt.imshow(display_image)

        label_index = np.argmax(labels[i])

        if hasattr(generator, 'class_indices'):
            idx_to_class = {v: k for k, v in generator.class_indices.items()}
            label_name = idx_to_class.get(label_index, 'Unknown')
        else:
            label_name = f"Index: {label_index}"

        plt.title(label_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

logging.info("\nDisplaying a batch of augmented training images:")
plot_augmented_images(train_generator)

logging.info("\n--- Plotting Training History ---")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(resnet101_history.history['accuracy'], label='Training Accuracy')
plt.plot(resnet101_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(resnet101_history.history['loss'], label='Training Loss')
plt.plot(resnet101_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

logging.info("\n--- Generating Classification Report and Confusion Matrix ---")

test_generator.reset()
Y_pred = resnet101.predict(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE), verbose=1)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

logging.info("\nClassification Report:")
report = classification_report(y_true_classes, y_pred_classes, target_names=class_labels, zero_division=0)
logging.info(report)

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(16, 14))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()


