import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import time

batch_size = 25
num_epochs = 5
learning_rate = 0.01
image_size = [28, 28]

def read_tfrecord(example):
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    
    label = []
    for val in list(df.columns)[2:]: label.append(example[val])

    return image, label

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord)
    
    return dataset

def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    return dataset

def tensorflow_to_pytorch(dataset):
    set = []
    for i, data in enumerate(dataset):
        images, labels = data
        images = torch.from_numpy(images.numpy()).permute(0, 3, 1, 2)
        labels = torch.from_numpy(labels.numpy())
        if images.size()[0] == batch_size:
            set.append([images, labels])
    return set

df = pd.read_csv('preprocessed_data.csv')

tfrlist = ['data/' + x for x in os.listdir('data')]
file_names = tf.io.gfile.glob(tfrlist)

all = list(range(len(file_names)))      # To decrease training time when testing modify this list to be shorter
train_index = random.sample(all, int(len(all) * 0.7))
test_and_validation_index = list(set(all) - set(train_index))
valid_index = random.sample(test_and_validation_index, int(len(test_and_validation_index) * 0.5))
text_index = list(set(test_and_validation_index) - set(valid_index))

train_file_names, valid_file_names, test_file_names = [file_names[index] for index in train_index], [file_names[index] for index in valid_index], [file_names[index] for index in text_index]

feature_description = {}
for elem in list(df.columns)[2:]:
    feature_description[elem] = tf.io.FixedLenFeature([], tf.int64)
feature_description['image'] = tf.io.FixedLenFeature([], tf.string)

print("Converting training data.")
train_dataset = tensorflow_to_pytorch(get_dataset(train_file_names))
print("Training data converted.")

print("Converting validation data.")
valid_dataset = tensorflow_to_pytorch(get_dataset(valid_file_names))
print("Validation data converted.")

print("Converting test data.")
test_dataset = tensorflow_to_pytorch(get_dataset(test_file_names))
print("Test data converted.")

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

print("Begin training...")

def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    for _, data in enumerate(data_loader):
        images, labels = data
        output = model(images)
        for i in range(batch_size):
            if torch.round(torch.sigmoid(output[0][i]).cpu()).eq(labels[0][i]).sum().item() == 14:
                correct += 1
            total += 1

    return correct / total

#Artifical Neural Network Architecture
class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.layer1 = nn.Linear(28*28*3, 300)
        self.layer2 = nn.Linear(300, 64)
        self.layer3 = nn.Linear(64, 14)
    def forward(self, img):
        flattened = img.reshape(-1, 28*28*3)
        activation1 = self.layer1(flattened)
        activation1 = F.relu(activation1)
        activation2 = self.layer2(activation1)
        activation2 = F.relu(activation2)
        activation3 = self.layer3(activation2)
        return activation3.unsqueeze(0)

net = ANNClassifier()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

train_accuracy = np.zeros(num_epochs)
validation_accuracy = np.zeros(num_epochs)
epochs = range(num_epochs)

start_time = time.time()
for epoch in range(num_epochs):
    for _, data in enumerate(train_loader):
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels.type_as(output))
        loss.backward()
        optimizer.step()
    train_accuracy[epoch] = get_accuracy(net, train_loader)
    validation_accuracy[epoch] = get_accuracy(net, valid_loader)
    print(f"Training accuracy: {train_accuracy[epoch]} Validation accuracy: {validation_accuracy[epoch]}")

print("Training complete.")

print(f"Test accuracy: {get_accuracy(net, test_loader)}")