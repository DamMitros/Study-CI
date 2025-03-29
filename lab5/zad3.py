import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

data = 'dogs-cats-mini'
img_width, img_height = 75 , 75
batch_size = 32

# Preprocess data

def create_dataframe(data):
    images = []
    labels = []
    
    cat_count = 0
    dog_count = 0
    
    # for filename in os.listdir(data):
    #     if filename.endswith('.jpg'):
    #         if filename.startswith('cat'):
    #             images.append(filename)
    #             labels.append(0)
    #             cat_count += 1
    #         elif filename.startswith('dog'):
    #             images.append(filename)
    #             labels.append(1)
    #             dog_count += 1
    
    all_files = os.listdir(data)
    
    cat_files = [f for f in all_files if f.startswith('cat') and f.endswith('.jpg')]
    for filename in cat_files[:1000]:  # Limit to 12,500 cat images
      images.append(filename)
      labels.append(0)
    
    # Process dog images
    dog_files = [f for f in all_files if f.startswith('dog') and f.endswith('.jpg')]
    for filename in dog_files[:1000]:  # Limit to 12,500 dog images
      images.append(filename)
      labels.append(1)

    print(f"Loaded {cat_count} cat images and {dog_count} dog images")
    df = pd.DataFrame({'filename': images, 'label': labels})
    return df

df = create_dataframe(data)
print(f'Images in {data}: {len(df)}')

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])

def load_image(dataframe, data):
  images = []
  labels = []

  for index, row in dataframe.iterrows():
    img_path = os.path.join(data, row['filename'])
    img = load_img(img_path, target_size=(img_width, img_height))
    img = img_to_array(img)/255.0

    images.append(img)
    labels.append(row['label'])
  return np.array(images), np.array(labels)

train_images, train_labels = load_image(train_df, data)
test_images, test_labels = load_image(test_df, data)

#b)

model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
  MaxPooling2D((2, 2)),

  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),

  Conv2D(128, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),

  Flatten(),
  Dense(128, activation='relu'),
  Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=batch_size, validation_split=0.2, callbacks=[history])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

predictions = model.predict(test_images)
predicted_labels = np.round(predictions).flatten()

cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], ['Cat', 'Dog'])
plt.yticks([0.5, 1.5], ['Cat', 'Dog'])
plt.title('Confusion Matrix')
plt.savefig('dog_cat_confusion_matrix.png')


