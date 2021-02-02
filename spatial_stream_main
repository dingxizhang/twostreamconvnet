import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.model_zoo as model_zoo
import math
import random
import numpy as np
from PIL import Image
import time
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = 1

# Define VGG-16
class VGG(nn.Module):

    def __init__(self, features, num_classes=102):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.9),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.9),
        )

        self.fc_action = nn.Linear(2048, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print('x:', x.size())
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc_action(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v,w,x,y in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=w, stride=x)
            if y == 1:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [(96, 7, 2, 1), ('M', 0, 0, 0), (256, 5, 2, 1), ('M', 0, 0, 0),
          (512, 3, 1, 0), (512, 3, 1, 0), (512, 3, 1, 0), ('M', 0, 0, 0)]
}

def rgb_vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['F']), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

# Build the network
# spatial_VGG = rgb_vgg16()


# Use GPU
device = torch.device("cuda")
print(device)

def get_error(scores, labels):

    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return 1-num_matches.float()

def get_image(file_list, count_list, index):
    path = 'ucf101_jpegs_256/jpegs_256'
    name_of_file = file_list[index]
    num_of_frame = count_list[index]
    # print(name_of_file, ' ', num_of_frame)
    chosen_frame = random.randint(1, num_of_frame)  # num of the chosen frame
    frame_path = 'frame%06d.jpg' % chosen_frame
    file_path = os.path.join(path, name_of_file)
    file_path = os.path.join(file_path, frame_path)  # file_path is the path of target frame

    train_data = Image.open(file_path)
    return train_data

#============= Data loading ==============

# import lists of training and testing dataset
f = open("trainlist01.txt")
trainlist = f.readlines()
f2 = open("testlist01.txt")
testlist = f2.readlines()
train_label = []
test_label = []
label2index = {}

# build a list to store the label in the same sequence of training data
# and change the trainlist to the path of folders for each data
# and build a dictionary to store labels' indices
for i in range(len(trainlist)):
    label = trainlist[i].split('/',1)[0]
    index = trainlist[i].split(' ',1)[1]
    if label not in label2index:
        label2index[label] = index
    train_label.append(index)
    train_label[-1] = int(train_label[-1])
    trainlist[i] = trainlist[i].split(' ',1)[0].split('/',1)[1].split('.',1)[0]
train_label = torch.tensor(train_label, dtype=torch.int64)
print("Train data loaded")

for i in range(len(testlist)):
    label = testlist[i].split('/',1)[0]
    test_label.append(label2index[label])
    test_label[-1] = int(test_label[-1])    # convert str to int
    testlist[i] = testlist[i].split('/',1)[1].split('.',1)[0]
test_label = torch.tensor(test_label, dtype=torch.int64)
print("Test  data loaded")

# import number of frames for each video
train_frame_count = pd.read_csv('train_frame_count.csv', header=None)    # num of frames of each video
train_frame_count = train_frame_count.iloc[:,0].values.tolist()
test_frame_count = pd.read_csv('test_frame_count.csv', header=None)
test_frame_count = test_frame_count.iloc[:,0].values.tolist()
path = 'ucf101_jpegs_256/jpegs_256'
print("Frame num imported")

#============= Pre-Training ===============

# Create neuron network
# net = torchvision.models.vgg16()
net = rgb_vgg16()
net = net.to(device)
print(net)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss().cuda()

# Initialize learning rate
my_lr = 0.01
bs = 128

# Define the evaluation function
def evaluate():

    running_error = 0
    num_batches = 0

    for i in range(0, len(testlist), bs):
        flag = 1
        transform1 = transforms.Compose([
            transforms.ToTensor(),
            ]
        )
        for index in range(i, i+bs):
            test_data = get_image(testlist, test_frame_count, index)
            input = transform1(test_data).numpy()
            input = np.expand_dims(input, axis=0)
            if flag == 1:
                minibatch_data = input.copy()
                flag = 0
            else:
                minibatch_data = np.append(minibatch_data, input, axis=0)

        inputs = torch.Tensor(minibatch_data).to(device)
        minibatch_label = np.array(test_label[i:i+bs])
        minibatch_label = torch.Tensor(minibatch_label).long().to(device)

        scores = net(inputs)

        error = get_error(scores, minibatch_label)

        running_error += error.item()

    print('error on test set =', running_error*100, 'percent')

#=============== Training =================
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
print("Start training")
start = time.time()
iterations = 0
for epoch in range(1, len(trainlist)):

    # set running quatities
    running_loss = 0
    running_error = 0
    num_batches = 0

    # create an optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)
    # update the learning rate w.r.t. iterations num
    if iterations == 50000:
        my_lr = 0.001
        optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)
    elif iterations == 70000:
        my_lr = 0.0001
        optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)

    # randomize the order of training set
    shuffled_indices = torch.randperm(len(trainlist))

    # training in one epoch
    for count in range(0, len(trainlist), bs):

        # set the gradients to zeros
        optimizer.zero_grad()

        # create a minibatch
        indices = shuffled_indices[count:count+bs]

        flag = 1
        for index in indices:
            # extract target frame
            train_data = get_image(trainlist, train_frame_count, index)

            myTransforms = transforms.Compose([
                transforms.RandomCrop(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                ]
            )
            input = myTransforms(train_data).numpy()
            input = np.expand_dims(input, axis=0)
            if flag == 1:
                minibatch_data = input.copy()
                flag = 0
            else:
                minibatch_data = np.append(minibatch_data, input, axis=0)

        # send to the device
        inputs = torch.Tensor(minibatch_data).to(device)
        minibatch_label = np.array(train_label[indices])
        minibatch_label = torch.Tensor(minibatch_label).long().to(device)

        # start tracking all operations
        inputs.requires_grad_()

        # forward inputs through the net
        scores = net(inputs)

        # compute the loss
        loss = criterion(scores, minibatch_label)

        # backward pass
        loss.backward()

        # one step of SGD
        optimizer.step()

        # compute stats
        # add the loss to the running loss
        running_loss += loss.detach().item()
        print(running_loss)

        # compute the error
        error = get_error(scores.detach(), minibatch_label)
        running_error += error.item()
        num_batches += 1
        iterations += 1

    # stats of the full training set
    total_loss = running_loss/num_batches
    total_error= running_error/num_batches
    clock = (time.time()-start)/60
    print('epoch=', epoch, '\t time=', clock, '\t lr=', my_lr, '\t loss=', running_loss,
          '\t error=', running_error*100, 'percent')
    evaluate()
    pri nt(' ')

# See how is the predictions
# idx = randint(0, len(testlist)-1)
