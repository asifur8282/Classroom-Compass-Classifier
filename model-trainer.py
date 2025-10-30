import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import load_img, img_to_array
import logging

# --- Setup and Initialization ---

# Initialize a logger for better debugging and info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


train_dir = 'train'
validation_dir = 'validate'


def get_trained_model():
    """
    Loads data, defines, compiles, and trains the Keras model.
    Returns the trained model.
    """
    
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    try:
        train_dataset = train_datagen.flow_from_directory(
            train_dir,
            target_size=(300, 300),
            batch_size=3,
            class_mode='binary'
        )

        validation_dataset = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(300, 300),
            batch_size=3,
            class_mode='binary'
        )
    except Exception as e:
        logger.error(f"Failed to load training/validation data. Please check your directory structure: {e}")
        # Re-raise the exception to stop the script
        raise

    model = tf.keras.models.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Second convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Third convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Fourth Convolutional layer
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten the results to feed into a dense neural network
        tf.keras.layers.Flatten(),
        
        # Hidden dense layer with 512 neurons
        tf.keras.layers.Dense(512, activation='relu'),

        # Output layer with a single neuron and a sigmoid activation for binary classification
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )

    logger.info("Starting model training. This will take a while...")

    model.fit(
        train_dataset,
        epochs=15,  
        steps_per_epoch=train_dataset.samples // train_dataset.batch_size,
        validation_data=validation_dataset,
        validation_steps=validation_dataset.samples // validation_dataset.batch_size
    )
    logger.info("Model training complete.")
    return model


if __name__ == "__main__":
    try:
        model = get_trained_model()


        model_save_path = "student_focus_classifier.keras"
        
        model.save(model_save_path)
        
        logger.info(f"Model successfully trained and saved to {model_save_path}")

    except Exception as e:
        logger.error(f"An error occurred during training or saving: {e}", exc_info=True)
        exit()

    logger.info("Script finished.")
