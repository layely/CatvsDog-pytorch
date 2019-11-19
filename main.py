import torch
import torch.nn as nn
import numpy as np
from torch.utils import data

from models import AlexNet
from dataset import DataSplit

import os


def stop():
    os._exit(0)


GPU = True
DATASET_DIR = os.environ["HOME"] + "/Myprojects/datasets/cat_vs_dog"

epochs = 10
batch_size = 8
num_iteration = 1000
input_size = (3, 80, 80)
output_size = 1
lr = 1e-4

# Datasplit
train, val, test = 70, 15, 15

if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset, val_dataset, test_dataset = DataSplit(DATASET_DIR, train, val, test).get_datasets()
train_generator = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             sampler=None, batch_sampler=None, num_workers=0,
                             collate_fn=None, pin_memory=False, drop_last=False,
                             timeout=0, worker_init_fn=None, multiprocessing_context=None)

model = AlexNet(input_size, output_size)
print(model)

if GPU:
    model = model.cuda()

loss_func = torch.nn.BCELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    accumulated_loss = 0
    for batch_x, batch_y in train_generator:
        # In pytorch, conv expect input shape to be in this
        # form: (batch_size, channels, height, weight).
        batch_x = batch_x.permute(0, 3, 1, 2)

        # move batches to correct device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # convert tensors type to float
        batch_x = batch_x.type(torch.cuda.FloatTensor)
        batch_y = batch_y.type(torch.cuda.FloatTensor)

        # Forward
        preds = model(batch_x)

        # compute loss
        loss = loss_func(preds, batch_y)
        accumulated_loss += loss.item()

        # zero gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # Step to update optimizer params
        optimizer.step()

    print("Epoch: {} -- -- total_loss: {}".format(epoch, accumulated_loss))
