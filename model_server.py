import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from fastapi import FastAPI, File, UploadFile
import uvicorn
import logging
from typing import List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MODEL_PATH = "student_focus_classifier.keras"
classifier = None

try:
    logger.info(f"Loading trained model from {MODEL_PATH}...")
    classifier = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
   
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}. Error: {e}", exc_info=True)
    logger.error("Please make sure the file exists and you have trained it first.")
    exit()



app = FastAPI()


def classify_images(image_data_list: List[bytes]):
    """
    Classifies a list of images as 'focusing' or 'distracted'.
    
    Args:
        image_data_list (List[bytes]): A list of raw image data as byte strings.
        
    Returns:
        dict: A dictionary with the count of each class.
    """
    
    images_to_predict = []
    
    for image_data in image_data_list:
        # Decode the image from bytes
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Failed to decode image from uploaded file.")
            continue

        # Resize and preprocess the image 
        img_resized = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
        x = img_to_array(img_resized)
        images_to_predict.append(np.expand_dims(x, axis=0))

    if not images_to_predict:
        logger.warning("No valid images were provided for prediction.")
        return {"focusing": 0, "distracted": 0}
        

    all_images = np.vstack(images_to_predict)
    normalized_images = all_images / 255.0

    
    predictions = classifier.predict(normalized_images, verbose=0)
    
    student_counts = {
        "distracted": 0,
        "focusing": 0
    }
    
    
    for prediction in predictions:
        if prediction[0] < 0.45:
            student_counts["distracted"] += 1
        else:
            student_counts["focusing"] += 1

    return student_counts


@app.post("/classify_students/")
async def classify_students_api(files: List[UploadFile] = File(...)):
    """
    API endpoint to classify a batch of students in uploaded images.
    
    Args:
        files (List[UploadFile]): The list of image files uploaded via the request.
        
    Returns:
        dict: A dictionary containing the final count of each class.
    """
    
    # Check if the model loaded correctly
    if classifier is None:
        logger.error("Model is not loaded. Cannot process request.")
        return {"prediction_status": "error", "message": "Model not loaded on server."}

    try:
        # Read all files into a list of bytes
        file_bytes_list = [await file.read() for file in files]
        
        logger.info(f"Received {len(files)} images for classification.")
        
        # Pass the bytes to the classification function
        student_counts = classify_images(file_bytes_list)
        
        logger.info(f"Total students classified: {sum(student_counts.values())}")
        logger.info(f"Final counts: Focusing={student_counts['focusing']}, Distracted={student_counts['distracted']}")
        
        return {
            "prediction_status": "success",
            "total_students": sum(student_counts.values()),
            "counts": student_counts
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return {"prediction_status": "error", "message": "Internal server error."}


if __name__ == "__main__":
    logger.info("Starting FastAPI server on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
