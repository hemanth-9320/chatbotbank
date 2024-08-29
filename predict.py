import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import Emotion

def age(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path)
    model = tf.keras.models.load_model(r'Flask\age_prediction_model6.h5')
    #predict age from image
    img_height, img_width = 224, 224  # Same as used during training
    preprocessed_image = preprocess_image(image_path, img_height, img_width)
    predicted_age = model.predict(preprocessed_image)
    predicted_age_class = np.argmax(predicted_age[0])
    return predicted_age_class + 1  # Adjusting for zero-based indexing



def gender(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path)
    model = tf.keras.models.load_model(r'Flask\gender_prediction_model3.h5')
    #predict gender from image
    img_height, img_width = 224, 224  # Same as used during training
    preprocessed_image = preprocess_image(image_path, img_height, img_width)
    predicted_gender = model.predict(preprocessed_image)
    return 'female' if predicted_gender[0][0] > 0.5 else 'male'


# Function to preprocess the image
def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Scale pixel values
    return img_array

def emotion(image_path):
    rslt = Emotion.predict(image_path)
    return rslt


img_path=r'uploads\test.jpg'
# print(age(img_path))
