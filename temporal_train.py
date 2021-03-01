import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.model_zoo as model_zoo
from torch.utils.data import dataset, DataLoader
import torch.utils.data as data
import math
import random
import numpy as np
from PIL import Image
import time
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# CUDA_VISIBLE_DEVICES = 1

# Define VGG-16
class VGG(nn.Module):

    def __init__(self, features, num_classes=101):
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
    in_channels = 20
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

def get_error(scores, labels):

    bs = scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return 1 - num_matches.float()/bs

def get_image(file_list, count_list, index):
    # u_path = '/media/shared/dingxi/temporal/tvl1_flow/u'
    # v_path = '/media/shared/dingxi/temporal/tvl1_flow/v'

    u_path = '/home/dingxi/temporal/tvl1_flow/u'
    v_path = '/home/dingxi/temporal/tvl1_flow/v'

    name_of_file = file_list[index]
    num_of_frame = count_list[index]
    # print(name_of_file, ' ', num_of_frame)
    # randomly choose a frame, then stack 4 images before it and 5 images after
    chosen_frame = random.randint(5, num_of_frame-6)  # num of the chosen frame
    flag = 1
    for i in range(chosen_frame-4, chosen_frame+6):
        frame_path = 'frame%06d.jpg' % i

        u_file_path = os.path.join(u_path, name_of_file)
        u_file_path = os.path.join(u_file_path, frame_path)  # file_path is the path of target frame

        v_file_path = os.path.join(v_path, name_of_file)
        v_file_path = os.path.join(v_file_path, frame_path)

        u_img = Image.open(u_file_path)
        v_img = Image.open(v_file_path)

        myTransforms = transforms.Compose([
            transforms.RandomCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        )

        u_frame = myTransforms(u_img).numpy()
        v_frame = myTransforms(v_img).numpy()

        # print('u_frame: ', u_frame.shape)

        if flag == 1:
            train_data = u_frame.copy()
            train_data = np.append(train_data, v_frame, axis=0)
            flag = 0
        else:
            train_data = np.append(train_data, u_frame, axis=0)
            train_data = np.append(train_data, v_frame, axis=0)

    return train_data

class myDataset(data.Dataset):

    def __init__(self, label_list, file_list, count_list, image_dir, T):
        self.label_list = label_list
        self.file_list = file_list
        self.count_list = count_list
        self.image_dir = image_dir
        self.len = len(self.file_list)
        self.T = T

    def __getitem__(self, i):
        index = i % self.len
        name_of_file = self.file_list[index]
        num_of_frame = self.count_list[index]

        dircs = ['u', 'v']
        if self.T == 0:
            # randomly choose a frame, then stack 4 images before it and 5 images after
            chosen_frame = random.randint(5, num_of_frame - 6)  # num of the chosen frame
            chosen_frames = range(chosen_frame - 4, chosen_frame + 6)
        else:
            chosen_frames = range(1, num_of_frame, int(num_of_frame / self.T))[:self.T]  # num of the chosen frame
        flag = 1

        for i in chosen_frames:
            frame_path = 'frame%06d.jpg' % i

            for dirc in dircs:

                file_path = os.path.join(self.image_dir, dirc)
                file_path = os.path.join(file_path, name_of_file)
                file_path = os.path.join(file_path, frame_path)

                img = Image.open(file_path)
                if self.T == 0:
                    myTransforms = transforms.Compose([
                        transforms.RandomCrop(size=(224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                    )
                else:
                    myTransforms = transforms.Compose([
                        transforms.CenterCrop(size=(224, 224)),
                        transforms.ToTensor(),
                    ]
                    )
                frame = myTransforms(img).numpy()

                if flag == 1:
                    train_data = frame.copy()
                    flag = 0
                else:
                    train_data = np.append(train_data, frame, axis=0)

        train_data = torch.Tensor(train_data)
        label = np.array([self.label_list[index]], dtype=int)
        return train_data, label

    def __len__(self):
        return len(self.file_list)

# Define a new evaluation function
def evaluate():

    running_error = 0
    num_batches = 0

    for count in range(0, len(testlist), bs):

        for minibatch_data, minibatch_label in test_data:

            minibatch_data =  minibatch_data.to(device)
            minibatch_label = torch.max(minibatch_label, 1)[0]
            minibatch_label= minibatch_label.to(device)

            scores = net(minibatch_data)

            error = get_error(scores, minibatch_label)

            running_error += error.item()

            num_batches += 1

            # release caches
            del minibatch_data, scores
            torch.cuda.empty_cache()

            # if num_batches % 10 == 0:
            #     print('epochs = ', count, 'error = ', (running_error/num_batches)*100, 'percent')

    total_error = running_error/num_batches
    print('error on test set =',  total_error * 100, 'percent')

#============= Data loading ==============

# import lists of training and testing dataset
f = open("/media/shared/dingxi/spatial/trainlist01.txt")
trainlist = f.readlines()
f2 = open("/media/shared/dingxi/spatial/testlist01.txt")
testlist = f2.readlines()
train_label = []
test_label = []
label2index = {}

# build a list to store the label in the same sequence of training data
# and change the trainlist to the path of folders for each data
# and build a dictionary to store labels' indices
for i in range(len(trainlist)):
    label = trainlist[i].split('/',1)[0]
    index = int(trainlist[i].split(' ',1)[1]) - 1
    if label not in label2index:
        label2index[label] = index
    train_label.append(index)
    trainlist[i] = trainlist[i].split(' ',1)[0].split('/',1)[1].split('.',1)[0]
train_label = torch.tensor(train_label, dtype=torch.int64)
print("Train data loaded")

for i in range(len(testlist)):
    label = testlist[i].split('/',1)[0]
    test_label.append(label2index[label])
    testlist[i] = testlist[i].split('/',1)[1].split('.',1)[0]
test_label = torch.tensor(test_label, dtype=torch.int64)
print("Test  data loaded")

# import number of frames for each video
train_frame_count = pd.read_csv('/media/shared/dingxi/spatial/train_frame_count.csv', header=None)    # num of frames of each video
train_frame_count = train_frame_count.iloc[:,0].values.tolist()
test_frame_count = pd.read_csv('/media/shared/dingxi/spatial/test_frame_count.csv', header=None)
test_frame_count = test_frame_count.iloc[:,0].values.tolist()
path = '/home/dingxi/temporal/tvl1_flow'
print("Frame num imported")

#============= Pre-Training ===============

# Use GPU
device = torch.device('cuda')
print(device)

# Create neuron network
# net = torchvision.models.vgg16()
net = rgb_vgg16()
net.load_state_dict(torch.load("t_params.pkl"))
net = net.to(device)
# print(net)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)

# Initialize learning rate
my_lr = 0.01
bs = 256

# Load the data
train_dataset = myDataset(train_label, trainlist, train_frame_count, path, 0)
train_data = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_dataset = myDataset(test_label, testlist, test_frame_count, path, 10)
test_data = DataLoader(test_dataset, batch_size=bs, shuffle=True)

#=============== Training =================
print("Start training")
start = time.time()
iterations = 47880
for epoch in range(1260, int(80000/(int(len(trainlist)/bs)))):

    # set running quatities
    running_loss = 0
    running_error = 0
    num_batches = 0

    # create an optimizer
    # update the learning rate w.r.t. iterations num
    if iterations >= 50000 and iterations < 70000:
        my_lr = 0.001
        optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)
    elif iterations >= 70000 and iterations < 80000:
        my_lr = 0.0001
        optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)
    elif iterations >= 80000:
        torch.save(net.state_dict(), 't_params.pkl')
        print('End of training')
        break
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)

    optimizer.zero_grad()

    # training in one epoch
    for minibatch_data, minibatch_label in train_data:

        # send them to the device
        minibatch_data = minibatch_data.to(device)
        # print(minibatch_label)
        minibatch_label = torch.max(minibatch_label, 1)[0]
        # print(minibatch_label)
        minibatch_label = minibatch_label.to(device)

        # start tracking all operations
        inputs = minibatch_data
        inputs.requires_grad_()

        # forward inputs through the net
        scores = net(inputs)

        # compute the loss
        loss = criterion(scores, minibatch_label)

        # # trick ENABLED
        # loss = loss/accumulation_steps
        # if( (num_batches + 1) % accumulation_steps ) == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     iterations += 1

        # trick DISABLED
        # set the gradients to zeros
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # one step of SGD
        optimizer.step()
        iterations += 1

        # compute stats
        # add the loss to the running loss
        running_loss += loss.detach().item()
        # print(running_loss)

        # compute the error
        error = get_error(scores.detach(), minibatch_label)
        running_error += error.item()
        num_batches += 1

        # release caches
        del inputs, scores, loss
        torch.cuda.empty_cache()

    # stats of the full training set
    total_loss = running_loss/num_batches
    total_error= running_error/num_batches
    clock = (time.time()-start)/60
    print('epoch=', epoch, '\t time=', clock, '\t lr=', my_lr, '\t loss=', running_loss,
          '\t error=', total_error * 100, '\t iterations=', iterations)
    if epoch % 10 == 0:
        # evaluate()
        torch.save(net.state_dict(), 't_params.pkl')

