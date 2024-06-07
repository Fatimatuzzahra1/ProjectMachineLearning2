import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the model
model = tf.keras.models.load_model('model_cnn.h5')

# Load label encoder from the existing dataset (optional if you have it saved, otherwise create it)
testing_csv = 'test_dataset.csv'  # Path to your testing CSV file
test_df = pd.read_csv(testing_csv)

# Initialize LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(test_df['Label'])


def predict_image(image_path, model, label_encoder):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at path {image_path}")
        return None
    image = cv2.resize(image, (48, 48))  # Adjust to the input size of your model
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)

    # Predict using the model
    prediction = np.argmax(model.predict(image))

    # Decode the prediction to get the label
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Display the image and the prediction
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted Label: {predicted_label}')
    plt.axis('off')
    plt.show()

    return predicted_label


# Path to the new image you want to test
new_image_path = 'img.jpg'  # Ganti dengan path gambar baru Anda

# Predict the new image
predicted_label = predict_image(new_image_path, model, label_encoder)
if predicted_label is not None:
    print(f'The predicted label for the new image is: {predicted_label}')
