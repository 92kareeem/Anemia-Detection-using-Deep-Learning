import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models

# Load and predict function
def predict_nail_anemia(image_path, model_path='nail_anemia_model.keras'):
    model = models.load_model(model_path)
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return "Anemia Detected" if prediction > 0.5 else "No Anemia"

# Example usage
if __name__ == "__main__":
    test_image_path = "No Anemia sample img.png"  # Replace with an actual image path
    result = predict_nail_anemia(test_image_path)
    print(result)
