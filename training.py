import torch
import numpy as np

def get_accuracy(model, data_loader, batch_size):
    correct = 0
    total = 0
    for _, data in enumerate(data_loader):
        images, labels = data
        output = model(images)
        for i in range(batch_size):
            # if torch.round(torch.sigmoid(output[0][i])).eq(labels[0][i]).sum().item() == labels.size()[2]:
            #     correct += 1
            # total += 1
            correct += torch.round(torch.sigmoid(output[0][i])).eq(labels[0][i]).sum().item()
            total += 14

    return correct / total

def train(net, train_loader, valid_loader, criterion, optimizer, num_epochs, batch_size):
    train_accuracy = np.zeros(num_epochs)
    validation_accuracy = np.zeros(num_epochs)
    epochs = range(num_epochs)

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
        train_accuracy[epoch] = get_accuracy(net, train_loader, batch_size)
        validation_accuracy[epoch] = get_accuracy(net, valid_loader, batch_size)
        print(f"Epoch: {epoch}, Training accuracy: {train_accuracy[epoch]}, Validation accuracy: {validation_accuracy[epoch]}")

    print("Training complete.")

    return train_accuracy, validation_accuracy, epochs