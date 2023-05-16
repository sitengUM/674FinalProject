import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import time


def get_data(path):
    print("get_data called {}".format(path))

    pc = pd.read_csv(path,
                     header=None,
                     delim_whitespace=True,
                     dtype=np.float32).values

    #points = pc[:, 0:3]
    #feat = pc[:, [4, 5, 6]]
    #intensity = pc[:, 3]

    input = np.array(pc, dtype=np.float32)
    num = len(input)
    print(num)
    reduced_input = input[:1000000]
    labels = pd.read_csv(path.replace(".txt", ".labels"),
                         header=None,
                         delim_whitespace=True,
                         dtype=np.float32).values
    labels = np.array(labels, dtype=np.float32).reshape((-1,))
    reduced_labels= labels[:1000000]
    reduced_input = torch.from_numpy(reduced_input)
    reduced_labels = torch.from_numpy(reduced_labels)
    print(reduced_input.shape, reduced_labels.shape)
    return reduced_input,reduced_labels
# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.leakyRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(7, 16,1)
        #self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 100,1)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 9)

    def forward(self, x):
        x = self.leakyRelu(self.conv1(x))
        x = self.leakyRelu(self.conv2(x))
        #x = x.view(-1, 16 * 4 * 4)
        x = self.leakyRelu(self.fc1(x))
        x = self.leakyRelu(self.fc2(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()
loss_fn = torch.nn.CrossEntropyLoss()

def train_one_epoch(epoch_index, training_loader):
    running_loss = 0.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        labels = labels.type(torch.LongTensor)
        inputs = inputs.reshape(7,100)
    # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        #print(outputs, labels)
        #print(outputs.shape)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

    # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss
    last_loss = running_loss / (i + 1)

    return last_loss

epoch_number = 0

EPOCHS = 20

best_vloss = 1000000
start_time = time.time()
train_data,train_label = get_data("input/point-cloud-segmentation/train/bildstein_station1_xyz_intensity_rgb.txt")
middle_time = time.time()
elapsed_time = middle_time - start_time
print("Elapsed time train: ", elapsed_time)
val_data,val_label =  get_data("input/point-cloud-segmentation/val/bildstein_station3_xyz_intensity_rgb.txt")
end_time = time.time()
elapsed_time = end_time - middle_time
print("Elapsed time val: ", elapsed_time)
train = torch.utils.data.TensorDataset(train_data, train_label)
val = torch.utils.data.TensorDataset(val_data, val_label)
training_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val, batch_size=100, shuffle=False)
#end_time_final = time.time()
#elapsed_time = end_time_final - end_time
#print("Elapsed time val: ", elapsed_time)


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number,training_loader)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.reshape(7, 100)
        vlabels = vlabels.type(torch.LongTensor)
        voutputs = model(vinputs)
        #print(voutputs.shape, vlabels)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} val {}'.format(avg_loss, avg_vloss))
    # Log the running loss averaged per batch
    # for both training and validation

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        print("yeah")
        #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #torch.save(model.state_dict(), model_path)

    epoch_number += 1

