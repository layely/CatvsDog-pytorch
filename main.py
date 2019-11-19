import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import os

from models import AlexNet
from dataset import DataSplit
from utils import accuracy

GPU = True
DATASET_DIR = os.environ["HOME"] + "/Myprojects/datasets/cat_vs_dog"

epochs = 10
batch_size = 32
input_size = (3, 80, 80)
output_size = 1
lr = 1e-4

# Datasplit
train, val, test = 70, 15, 15

if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset, val_dataset, test_dataset = DataSplit(DATASET_DIR, train, val, test).get_datasets(device)
train_generator = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_generator = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = AlexNet(input_size, output_size)
print(model)

if GPU:
    model = model.cuda()

loss_func = torch.nn.BCELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    # Train
    accumulated_train_loss = 0
    for batch_x, batch_y in train_generator:
        # Forward
        preds = model(batch_x)

        # compute loss
        loss = loss_func(preds, batch_y)
        accumulated_train_loss += loss.item()

        # zero gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # Step to update optimizer params
        optimizer.step()

    # Validation
    accumulated_val_loss = 0
    for batch_x, batch_y in val_generator:
        # Forward
        preds = model(batch_x)

        # compute loss
        loss = loss_func(preds, batch_y)
        accumulated_val_loss += loss.item()

    print("Epoch: {} -- -- train loss: {}, val loss: {}".format(epoch, accumulated_train_loss, accumulated_val_loss))

# Evaluate model
test_generator = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
accumulated_test_loss = 0
metrics = []
for batch_x, batch_y in test_generator:
    # Forward
    preds = model(batch_x)
    metrics.append(accuracy(preds, batch_y))

    # compute loss
    loss = loss_func(preds, batch_y)
    accumulated_test_loss += loss.item()

print("Accuracy on test data: ", sum(metrics) / len(metrics))
