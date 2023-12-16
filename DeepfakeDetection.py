import opendatasets as od
import os
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTConfig


from tqdm import tqdm

od.download_kaggle_dataset("manjilkarki/deepfake-and-real-images",'')

# Directories setup
base_dir = 'deepfake-and-real-images/Dataset'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

# Model configuration
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

adamw_optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

# Model compilation
model.compile(optimizer=adamw_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Number of batches to draw from the generator per epoch
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50,  # Number of batches to draw from the validation generator
    verbose=2
)


# Model evaluation
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test accuracy:', test_accuracy)

#Saving the model
#model.save('deep_fake_detector_model.h5')



print("Now running Transformers model")
# Defining transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading datasets
train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=validation_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

# Creating data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# Loading the configuration of the model
config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=2)

# Creating the model with the custom configuration
# This initializes a new classifier layer with the correct number of labels
model = ViTForImageClassification(config)

# Alternatively, can load the pre-trained model without its head
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)


# Replacing the classifier head
model.classifier = nn.Linear(model.config.hidden_size, 2)

# Model set up with a binary classifier at the end

model.config.id2label = {1: 'Real', 0: 'Fake'}
model.config.label2id = {'Real': 1, 'Fake': 0}

# Setting up training (optimizer, loss function, etc.)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()


# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Number of epochs
num_epochs = 10

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f"Training Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

    # Saving the model if it has better performance or periodically
    # torch.save(model.state_dict(), 'model_epoch_{epoch}.pth')


# Testing Phase
model.eval()  
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # No gradients needed for testing phase
    for images, labels in tqdm(test_loader, desc="Testing Phase"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {test_accuracy:.2f}%")


