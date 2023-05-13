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

    points = pc[:, 0:3]
    feat = pc[:, [4, 5, 6]]
    intensity = pc[:, 3]

    points = np.array(points, dtype=np.float32)
    feat = np.array(feat, dtype=np.float32)
    intensity = np.array(intensity, dtype=np.float32)

    labels = pd.read_csv(path.replace(".txt", ".labels"),
                         header=None,
                         delim_whitespace=True,
                         dtype=np.int32).values
    labels = np.array(labels, dtype=np.int32).reshape((-1,))

    data = {
        'point': points,
        'feat': feat,
        'intensity': intensity,
        'label': labels
    }

    return data
# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()
loss_fn = torch.nn.CrossEntropyLoss()

def train_one_epoch(epoch_index, training_loader):
    running_loss = 0.
    last_loss = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data['points']. data['label']

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1

            running_loss = 0.

    return last_loss

epoch_number = 0

EPOCHS = 5

best_vloss = 1000000
start_time = time.time()
train_data = get_data("input/point-cloud-segmentation/train/mock_data.txt")
middle_time = time.time()
elapsed_time = middle_time - start_time
print("Elapsed time train: ", elapsed_time)
val_data =  get_data("input/point-cloud-segmentation/val/mock_data.txt")
end_time = time.time()
elapsed_time = end_time - middle_time
print("Elapsed time val: ", elapsed_time)
training_loader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_data, batch_size=40, shuffle=False)
end_time_final = time.time()
elapsed_time = end_time_final - end_time
print("Elapsed time val: ", elapsed_time)


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number,training_loader)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, data in enumerate(validation_loader):
        vinputs, vlabels = data['points']. data['label']
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i+1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        print("yeah")
        #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #torch.save(model.state_dict(), model_path)

    epoch_number += 1

