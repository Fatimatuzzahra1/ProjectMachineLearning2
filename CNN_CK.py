import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

#Load csv file containing image paths and labels
csv_path = "train_dataset.csv"
df = pd.read_csv(csv_path)

#split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#create image data generator
train_datagen = ImageDataGenerator(rescale=1./255)

#train_datagen = ImageDataGenerator(
#   rescale=1./255,
#   rotation_range=20,
#   width_shift_range=0.2,
#   height_shift_range=0.2,
#   shear_range=0.2,
#   zoom_range=0.2,
#   horizontal_flip=true,
#   fill_mode='nearest'
#)

test_datagen = ImageDataGenerator(rescale=1./255)

# specify batch size and image size
batch_size = 16
img_size = (48, 48)

#create data generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='Link',
    y_col='Label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='Link',
    y_col='Label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

model = models.Sequential()

#add convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(7, activation='softmax')) #num_classes is the number

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

num_epochs = 100

history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=test_generator
)

model.summary()

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

loss = history.history['loss']
accuracy = history.history['accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Get the number of epochs
epochs = range(1, len(loss) + 1)

# Plotting the loss
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the accuracy
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('model_cnn.h5')


