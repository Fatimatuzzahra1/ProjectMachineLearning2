import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('model_cnn.h5')  # Ganti dengan nama file model Anda

testing_csv = 'test_dataset.csv'

test_df = pd.read_csv(testing_csv)

# Initialize LabelEncoder
label_encoder = LabelEncoder()
test_df['Label'] = label_encoder.fit_transform(test_df['Label'])

# Mendefinisikan list untuk menyimpan prediksi dan label sebenarnya
predicted_labels = []
true_labels = []

# Loop melalui setiap folder label
for index, row in test_df.iterrows():
    image_path = row['Link']  # Assuming 'Link' column contains the image paths
    true_label = int(row['Label'])  # Assuming 'Label' column contains the true labels

    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (48, 48))  # Adjust to the input size of your model
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)

    # Predict using the model
    prediction = np.argmax(model.predict(image))

    # Store the predicted and true labels
    predicted_labels.append(prediction)
    true_labels.append(true_label)

# Menghitung confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Visualisasi confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

# Membuat laporan confusion
report = classification_report(true_labels, predicted_labels)

# Mencetak laporan confusion
print("Classification Report:")
print(report)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Menghitung akurasi
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Menghitung presisi
precision = precision_score(true_labels, predicted_labels, average='weighted')
print("Precision:", precision)

# Menghitung recall
recall = recall_score(true_labels, predicted_labels, average='weighted')
print("Recall:", recall)

# Menghitung f1-score
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print("F1-Score:", f1)
