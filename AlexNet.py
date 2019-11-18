import torch
import torch.nn as nn
import numpy as np

GPU = True

class AlexNet(nn.Module):
    def __init__(self, image_shape, num_classes):
        super(AlexNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # Compute visual encoder output dim
        features_size = self.get_output_dim(self.feature_extractor, image_shape)
        self.classifier = nn.Sequential(
            nn.Linear(features_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        prediction = self.classifier(features)
        return prediction

    def get_output_dim(self, model, image_dim):
        return model(torch.rand(1, *(image_dim))).data.view(1, -1).size(1)

epochs = 2
batch_size = 32
num_iteration = 1000
input_size = (3, 80, 80)
output_size = 1
lr = 1e-4

if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

inputs = torch.randn((batch_size,) + (input_size), device=device)
outputs = torch.randn((batch_size, output_size), device=device)

print("Inputs shape: ", inputs.shape)
print("Outputs shape: ", outputs.shape)

# outputs_cpu = outputs.cpu()
# outputs_np = outputs_cpu.numpy()
# print(outputs_np[0, :].shape)

model = AlexNet(input_size, output_size)
print(model)

if GPU:
    model = model.cuda()

loss_func = torch.nn.BCELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    for i in range(num_iteration):
        # Forward
        preds = model(inputs)

        # compute loss
        loss = loss_func(preds, outputs)
        if i % 100 == 99:
            print("Iteration: {} -- -- loss: {}".format(i, loss.item()))

        # zero gradients
        optimizer.zero_grad()

        # backward
        loss.backward()

        # Step to update optimizer params
        optimizer.step()