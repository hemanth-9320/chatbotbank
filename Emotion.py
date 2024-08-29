# Built-in dependencies
import os

# 3rd party dependencies
import numpy as np
import cv2

# Project dependencies
from deepface.commons import package_utils, folder_utils
from deepface.models.Demography import Demography

# Import TensorFlow and Keras based on the TensorFlow version
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

# Labels for the emotions that can be detected by the model
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Define the EmotionClient class
class EmotionClient(Demography):
    def __init__(self):
        self.model = load_model()
        self.model_name = "Emotion"

    def predict(self, img: np.ndarray) -> np.ndarray:
        # Preprocess the input image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_gray = cv2.resize(img_gray, (48, 48))  # Resize to 48x48
        img_gray = np.expand_dims(img_gray, axis=0)  # Add batch dimension
        img_gray = np.expand_dims(img_gray, axis=-1)  # Add channel dimension

        # Get emotion predictions
        emotion_predictions = self.model(img_gray, training=False).numpy()[0, :]

        return emotion_predictions

def load_model(url="facial_expression_model_weights.h5") -> Sequential:
    num_classes = 7

    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    # Load weights
    home = folder_utils.get_deepface_home()
    output = os.path.join(home, ".deepface/weights/facial_expression_model_weights.h5")

    if not os.path.exists(output):
        # If weights are not found, download them using gdown
        import gdown
        gdown.download(url, output, quiet=False)

    model.load_weights(output)

    return model

# Main execution block

def predict():
    if True:# __name__ == "__main__":
        # Initialize the EmotionClient
        emotion_client = EmotionClient()

        # Load an image
        img_path="uploads\captured_image.jpg"
        # img_path = "img4.jpg"
        image = cv2.imread(img_path)

        # Predict emotion
        emotion_predictions = emotion_client.predict(image)

        # Find the highest probability emotion
        emotion_label = labels[np.argmax(emotion_predictions)]
        emotion_probability = np.max(emotion_predictions)
        print("Emotion is---------",emotion_label)
        # print(f"Detected emotion: {emotion_label} with confidence {emotion_probability:.2f}")
        return emotion_label

exp=predict()
print(type(exp))
