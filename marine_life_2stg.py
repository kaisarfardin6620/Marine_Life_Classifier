import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
import os
import random
from PIL import Image
import logging
import math
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging to display INFO messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def set_random_seed(seed=42):
    """Sets random seeds for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set the random seed for reproducibility
set_random_seed()

# Define image and training parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS_FEATURE_EXTRACTION = 25
EPOCHS_FINE_TUNING = 15
NUM_CLASSES = 22 # Assuming 22 classes based on the dataset
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=12, # Number of epochs with no improvement after which training will be stopped.
    verbose=1,
    mode='max', # Stop when validation accuracy stops increasing
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
)

# Define ModelCheckpoint callback to save the best model during feature extraction
checkpoint_feature_extraction = ModelCheckpoint(
    'best_marine_life_model_feature_extraction.keras', # File path to save the model
    monitor='val_accuracy', # Metric to monitor
    save_best_only=True, # Save only the best model
    mode='max', # Save when validation accuracy is maximized
    verbose=1
)

# Callbacks for feature extraction phase
callbacks_feature_extraction = [early_stopping, checkpoint_feature_extraction]

# Base path for the dataset
base_path = 'marine_life_split'

def create_dataframe_from_folder(base_dir):
    """
    Creates a pandas DataFrame from image files organized in a folder structure.
    Assumes structure: base_dir/split_name/class_name/image.jpg
    """
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
                    # Check for common image file extensions
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        full_path = os.path.join(class_path, file)
                        relative_path = os.path.relpath(full_path, base_dir)
                        filepaths.append(relative_path)
                        labels.append(class_name)
                        data_sets.append(split_name)

    # Create DataFrame
    new_df = pd.DataFrame({
        'filepaths': filepaths,
        'labels': labels,
        'image_path': [os.path.join(base_dir, fp) for fp in filepaths],
        'data set': data_sets
    })
    return new_df

logging.info(f"Scanning '{base_path}' to create initial DataFrame from folder structure...")
df = create_dataframe_from_folder(base_path)

# Separate DataFrames for train, test, and validation sets
train_df_original = df[df['data set'] == 'train'].copy().rename(columns={'labels': 'label'})
test_df_original = df[df['data set'] == 'test'].copy().rename(columns={'labels': 'label'})
validation_df_original = df[df['data set'] == 'val'].copy().rename(columns={'labels': 'label'})

# --- ImageDataGenerator Setup ---
# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rotation_range=25, # Rotate images by up to 25 degrees
    width_shift_range=0.25, # Shift images horizontally by up to 25% of width
    height_shift_range=0.25, # Shift images vertically by up to 25% of height
    shear_range=0.15, # Shear transformations
    zoom_range=0.25, # Zoom in/out by up to 25%
    horizontal_flip=True, # Randomly flip images horizontally
    vertical_flip=True, # Randomly flip images vertically
    brightness_range=[0.7, 1.3], # Adjust brightness
    channel_shift_range=0.1, # Randomly shift channel values
    fill_mode='nearest', # Fill newly created pixels with the nearest available pixel
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input # Preprocess input for MobileNetV2
)

# No augmentation for validation and test data, only preprocessing
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Create data generators from DataFrames
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df_original,
    x_col='image_path', # Column containing image file paths
    y_col='label', # Column containing labels
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Target image size
    batch_size=BATCH_SIZE, # Batch size for training
    class_mode='categorical', # Labels are one-hot encoded
    shuffle=True # Shuffle training data
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=validation_df_original,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Do not shuffle validation data
)

test_generator = val_datagen.flow_from_dataframe(
    dataframe=test_df_original,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Do not shuffle test data
)

def create_enhanced_model(base_model_class, input_shape, num_classes, dropout_rate=0.5):
    """
    Creates a transfer learning model using a pre-trained base model (e.g., MobileNetV2)
    and adds custom dense layers for classification.
    """
    # Load the pre-trained base model without the top (classification) layer
    base_model = base_model_class(
        weights='imagenet',
        include_top=False, # Exclude the ImageNet classification head
        input_shape=input_shape # Input shape of the images
    )

    # Freeze the base model layers during feature extraction
    base_model.trainable = False

    # Build the sequential model
    model = Sequential([
        base_model, # The pre-trained convolutional base
        GlobalAveragePooling2D(), # Global average pooling to flatten the features

        # Custom dense layers for classification with L2 regularization, Batch Normalization, LeakyReLU, and Dropout
        Dense(1024, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.01),
        Dropout(dropout_rate),

        Dense(512, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.01),
        Dropout(dropout_rate * 0.8), # Slightly less dropout for deeper layers

        Dense(256, kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.01),
        Dropout(dropout_rate * 0.6), # Even less dropout

        Dense(num_classes, activation='softmax') # Output layer with softmax for multi-class classification
    ])

    return model

def create_lr_schedule(initial_lr, total_steps):
    """
    Creates a Cosine Decay learning rate schedule.
    """
    return CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=0.01 # Minimum learning rate as a fraction of initial_learning_rate
    )

logging.info("=== PHASE 1: FEATURE EXTRACTION TRAINING ===")

# Best learning rate determined from previous experiments or initial testing
best_lr = 0.0002

# Create the model for feature extraction
model = create_enhanced_model(MobileNetV2, input_shape, NUM_CLASSES)

# Calculate total steps for the learning rate schedule in Phase 1
total_steps_phase1 = math.ceil(train_generator.samples / BATCH_SIZE) * EPOCHS_FEATURE_EXTRACTION
lr_schedule_phase1 = create_lr_schedule(best_lr, total_steps_phase1)

# Compile the model for feature extraction
optimizer_phase1 = Adam(learning_rate=lr_schedule_phase1)
model.compile(
    optimizer=optimizer_phase1,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # Use label smoothing to prevent overfitting
    metrics=['accuracy']
)

model.summary() # Print model summary

logging.info("Starting feature extraction training...")
history_phase1 = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    epochs=EPOCHS_FEATURE_EXTRACTION,
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / BATCH_SIZE),
    callbacks=callbacks_feature_extraction # Use callbacks defined for feature extraction
)

logging.info("\n=== PHASE 2: FINE-TUNING ===")

# Unfreeze the base model for fine-tuning
base_model = model.layers[0]
base_model.trainable = True

# Define how many layers from the base model to fine-tune
# Fine-tune from this layer onwards; layers before this will remain frozen.
fine_tune_at = 100 # Example: unfreeze the last 28 layers of MobileNetV2 (total layers ~128)

# Freeze layers before 'fine_tune_at'
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Define learning rate for fine-tuning (typically lower than feature extraction)
fine_tune_lr = best_lr / 10 # Reduce learning rate for fine-tuning

# Calculate total steps for the learning rate schedule in Phase 2
total_steps_phase2 = math.ceil(train_generator.samples / BATCH_SIZE) * EPOCHS_FINE_TUNING
lr_schedule_fine_tune = create_lr_schedule(fine_tune_lr, total_steps_phase2)

# Re-compile the model for fine-tuning with the new learning rate and unfrozen layers
model.compile(
    optimizer=Adam(learning_rate=lr_schedule_fine_tune),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Callbacks for fine-tuning phase
fine_tune_callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, verbose=1, mode='max', restore_best_weights=True),
    ModelCheckpoint('best_marine_life_finetuned.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

logging.info("Starting fine-tuning...")
history_phase2 = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    epochs=EPOCHS_FINE_TUNING,
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / BATCH_SIZE),
    callbacks=fine_tune_callbacks,
    initial_epoch=len(history_phase1.history['loss']) # Start fine-tuning epochs from where feature extraction left off
)

def combine_histories(hist1, hist2):
    """
    Combines two Keras History objects into a single dictionary.
    Handles cases where hist2 might be empty or incomplete.
    """
    combined = {}
    # Iterate over keys from the first history object (which should always be complete)
    for key in hist1.history.keys():
        combined[key] = hist1.history[key]
        # If the key exists in the second history and its list is not empty, extend it
        if key in hist2.history and hist2.history[key]:
            combined[key].extend(hist2.history[key])
        else:
            # If a key is missing or empty in hist2, pad with the last value from hist1
            # This is a common approach for plotting, assuming no new data for that key
            # If you need strict padding, consider np.full or similar.
            if hist1.history[key]: # Ensure hist1 has data for this key
                last_val = hist1.history[key][-1]
                # Pad with 'nan' or the last value if hist2 did not run any epochs
                # For plotting, repeating the last value is often acceptable
                combined[key].extend([last_val] * len(hist2.history.get(key, [])))
            else:
                combined[key].extend([]) # If hist1 was also empty for some reason, keep it empty

    # Additionally, check for 'lr' in hist2 if it wasn't in hist1 initially
    # This might happen if LR schedule is only applied in phase 2 and not captured in hist1's keys
    if 'lr' in hist2.history and 'lr' not in combined:
        combined['lr'] = [lr_schedule_phase1(i).numpy() for i in range(len(hist1.history['loss']))] + hist2.history['lr']
    elif 'lr' in hist2.history and 'lr' in combined:
         # Ensure the combined lr also includes the second phase's LR correctly
         # The current structure of combined[key].extend(hist2.history[key]) should handle this if 'lr' is present in hist1
         pass # Already handled by the loop if 'lr' was in hist1.history.keys()

    return combined

# Combine the training histories from both phases
combined_history = combine_histories(history_phase1, history_phase2)

logging.info("\n=== FINAL EVALUATION ===")
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(
    test_generator,
    steps=math.ceil(test_generator.samples / BATCH_SIZE)
)

logging.info(f'Final Test Accuracy: {test_accuracy:.4f}')
logging.info(f'Final Test Loss: {test_loss:.4f}')

def plot_training_history(history, phase1_epochs):
    """
    Plots the training and validation accuracy/loss and learning rate over epochs.
    Highlights the start of the fine-tuning phase.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['accuracy']) + 1)

    # Plot Model Accuracy
    axes[0, 0].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
    axes[0, 0].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    axes[0, 0].axvline(x=phase1_epochs, color='g', linestyle='--', label='Fine-tuning Starts')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot Model Loss
    axes[0, 1].plot(epochs, history['loss'], 'b-', label='Training Loss')
    axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 1].axvline(x=phase1_epochs, color='g', linestyle='--', label='Fine-tuning Starts')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot Learning Rate Schedule (if 'lr' is available in history)
    if 'lr' in history:
        # Ensure lr values are floats for plotting
        lr_values = [float(val) for val in history['lr']]
        axes[1, 0].plot(epochs, lr_values, 'g-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log') # Use log scale for learning rate
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        logging.warning("Learning rate (lr) not found in history for plotting.")

    # Plot Generalization Gap (Validation Accuracy - Training Accuracy)
    accuracy_diff = np.array(history['val_accuracy']) - np.array(history['accuracy'])
    axes[1, 1].plot(epochs, accuracy_diff, 'purple', label='Val - Train Accuracy')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3) # Line at y=0 for reference
    axes[1, 1].axvline(x=phase1_epochs, color='g', linestyle='--', label='Fine-tuning Starts')
    axes[1, 1].set_title('Generalization Gap')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout() # Adjust layout to prevent overlapping
    plt.show()

# Plot the combined training history
plot_training_history(combined_history, EPOCHS_FEATURE_EXTRACTION)

def analyze_model_performance(model, test_generator):
    """
    Performs detailed performance analysis including classification report,
    confusion matrix, and per-class accuracy.
    """
    logging.info("\n=== DETAILED PERFORMANCE ANALYSIS ===")

    # Reset the test generator to ensure predictions start from the beginning
    test_generator.reset()
    # Get predictions for the test set
    predictions = model.predict(test_generator, steps=math.ceil(test_generator.samples / BATCH_SIZE))
    # Get the predicted class indices
    predicted_classes = np.argmax(predictions, axis=1)

    # Get the true class indices
    true_classes = test_generator.classes
    # Get class labels from the generator's class_indices
    class_labels = list(test_generator.class_indices.keys())

    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    # Plot Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Calculate Per-Class Accuracy
    # Handle division by zero if a class has no true instances
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_accuracy = np.true_divide(cm.diagonal(), cm.sum(axis=1))
        per_class_accuracy[cm.sum(axis=1) == 0] = np.nan # Set accuracy to NaN for classes with no true samples

    accuracy_df = pd.DataFrame({
        'Class': class_labels,
        'Accuracy': per_class_accuracy
    }).sort_values('Accuracy')

    # Plot Per-Class Accuracy
    plt.figure(figsize=(12, 8))
    sns.barplot(data=accuracy_df.dropna(), x='Accuracy', y='Class', palette='viridis') # Drop NaN for plotting
    plt.title('Per-Class Accuracy')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.show()

    return accuracy_df

# Analyze model performance on the test set
performance_df = analyze_model_performance(model, test_generator)

# Save the final trained model
model.save('final_marine_life_model.keras')
logging.info("Final model saved as 'final_marine_life_model.keras'")

logging.info(f"\n=== TRAINING COMPLETE ===")
logging.info(f"Best model saved during feature extraction: 'best_marine_life_model_feature_extraction.keras'")
logging.info(f"Best model saved during fine-tuning: 'best_marine_life_finetuned.keras'")
logging.info(f"Final model saved: 'final_marine_life_model.keras'")
logging.info(f"Final Test Accuracy: {test_accuracy:.4f}")
logging.info(f"Final Test Loss: {test_loss:.4f}")
