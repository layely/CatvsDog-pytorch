import torch
import numpy as np

GPU = True

class LayeNet(torch.nn.Module):
    def __init__(self, tensors_in, tensors_middle, tensors_out):
        # Init module
        super(LayeNet, self).__init__()

        # Create a model with 3 FC (one hidden layer)
        self.linear1 = torch.nn.Linear(tensors_in, tensors_middle)
        self.linear2 = torch.nn.Linear(tensors_middle, tensors_middle)
        self.linear3 = torch.nn.Linear(tensors_middle, tensors_out)

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(output)
        output = self.linear3(output)
        return output


batch_size = 64
num_iteration = 1000
input_size = 512
laten_dim = 32
output_size = 5
lr = 1e-4

if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

inputs = torch.randn((batch_size, input_size), device=device)
outputs = torch.randn((batch_size, output_size), device=device)

outputs_cpu = outputs.cpu()
outputs_np = outputs_cpu.numpy()
print(outputs_np[0, :].shape)

model = LayeNet(input_size, laten_dim, output_size)
if GPU:
    model = model.cuda()

loss_func = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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