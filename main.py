import preprocessing as p
import training as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

batch_size = 50
num_epochs = 5
learning_rate = 0.001
image_size = [100, 100]

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
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

class SimpleCNN(nn.Module):
    def __init__(self, kernel_size = 5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, kernel_size)
        self.conv3 = nn.Conv2d(3, 3, kernel_size)
        self.conv_to_fc = 48
        self.fc1 = nn.Linear(self.conv_to_fc, 32)
        self.fc2 = nn.Linear(32, 14)    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.squeeze(0))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batch_size, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(0)

class DeepCNN(nn.Module):
    def __init__(self, kernel_size = 5):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_to_fc = 20
        self.fc1 = nn.Linear(self.conv_to_fc, 32)
        self.fc2 = nn.Linear(32, 14)

    def forward(self, x):
        x = x.squeeze(0)
        x = self.pool(F.relu(self.conv1(x)))
        for i in range(50):
            x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batch_size, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(0)

def get_image_size():
    return image_size

def main():
    train_dataset, valid_dataset, test_dataset = p.get_datasets(batch_size, sample=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    net = DeepCNN()
    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

    train_accuracy, validation_accuracy, epochs = t.train(net, train_loader, valid_loader, criterion, optimizer, num_epochs, batch_size)

    plt.plot(epochs, train_accuracy)
    plt.title("Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.show()

    plt.plot(epochs, validation_accuracy)
    plt.title("Validation Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.show()

    print(f"Test accuracy: {t.get_accuracy(net, test_loader, batch_size)}")

if __name__ == "__main__":
    main()