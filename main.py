import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import os

from models import AlexNet
from dataset import DataSplit
from utils import accuracy, save_checkpoint, load_checkpoint, TBManager

# Set miscellaneous parameters
GPU = True
DATASET_DIR = os.environ["HOME"] + "/Myprojects/datasets/cat_vs_dog"
CHECKPOINT_FREQUENCY = 100
RESUME_TRAINING = False

# Set hyperparameters
epochs = 10
batch_size = 32
input_size = (3, 80, 80)
output_size = 1
lr = 1e-4
momentum = 0.9
opt = torch.optim.SGD

# Datasplit
train, val, test = 70, 15, 15

if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset, val_dataset, test_dataset = DataSplit(
    DATASET_DIR, train, val, test, input_shape=input_size[1:]).get_datasets(device)
train_generator = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_generator = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

# Initialize tensorboard
tensorboard = TBManager()

# Visulaize some images of the training set
cat_samples = train_dataset.images[:10] # The first images on the list are cats
dog_samples = train_dataset.images[-10:] # The last images on the list are dogs
for i in range(cat_samples.shape[0]):
    tensorboard.add_images("Cat/{}".format(i), None, cat_samples[i:i+1])
    tensorboard.add_images("Dog/{}".format(i), None, dog_samples[i:i+1])

model = AlexNet(input_size, output_size);
cur_epoch = 0

if RESUME_TRAINING:
    cur_epoch, optimizer_states_dict, loss = load_checkpoint(model)
    cur_epoch += 1

model = model.to(device)
print(model)

loss_func = torch.nn.BCELoss(reduction="mean")
optimizer = opt(model.parameters(), lr=lr, momentum=momentum)
if RESUME_TRAINING:
    optimizer.load_state_dict(optimizer_states_dict)

# Train

for epoch in range(cur_epoch, epochs):
    accumulated_train_loss = []
    # Set model in trainng mode
    model.train()
    iteration = 0
    for batch_x, batch_y in train_generator:
        # Forward
        preds = model(batch_x)

        # compute loss
        loss = loss_func(preds, batch_y)
        accumulated_train_loss.append(loss.item())

        # zero gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # Step to update optimizer params
        optimizer.step()

        iteration += 1

    # Validation
    # Set mode in inference mode
    model.eval()
    accumulated_val_loss = []
    for batch_x, batch_y in val_generator:
        # Forward
        preds = model(batch_x)

        # compute loss
        loss = loss_func(preds, batch_y)
        accumulated_val_loss.append(loss.item())

    train_loss = sum(accumulated_train_loss) / len(accumulated_train_loss)
    val_loss = sum(accumulated_val_loss) / len(accumulated_val_loss)
    print("Epoch: {} -- -- train loss: {}, val loss: {}".format(epoch,
                                                                train_loss, val_loss))
    tensorboard.add_scalar("Loss/Train", train_loss, epoch)
    tensorboard.add_scalar("Loss/Validation", val_loss, epoch)

    # Save checkpoint, if applicable
    if CHECKPOINT_FREQUENCY > 0 and (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
        save_checkpoint(model, optimizer, epoch, loss)


# Evaluate model
model.eval()
test_generator = data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)
accumulated_test_loss = []
metrics = []
for batch_x, batch_y in test_generator:
    # Forward
    preds = model(batch_x)
    metrics.append(accuracy(preds, batch_y))

    # compute loss
    loss = loss_func(preds, batch_y)
    accumulated_test_loss.append(loss.item())

print("Accuracy on test data: ", sum(metrics) / len(metrics))
