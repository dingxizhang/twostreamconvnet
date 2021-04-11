import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.model_zoo as model_zoo
from torch.utils.data import dataset, DataLoader
from torch.utils.checkpoint import checkpoint_sequential
import torch.utils.data as data
import math
import random
import numpy as np
from PIL import Image
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# from apex import amp

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# CUDA_VISIBLE_DEVICES = 1

# Define VGG-16
class VGG(nn.Module):

    def __init__(self, features, num_classes=101):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.9),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.9),
        )

        self.fc_action = nn.Linear(2048, num_classes)
        self._initialize_weights()

    def forward(self, x):
        # # use checkpoint
        # num_segments = 2
        # x = checkpoint_sequential(self.features, num_segments, x)

        x = self.features(x)
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
                layers += [conv2d, nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                # layers += [conv2d, nn.ReLU(inplace=True)]
                # layers += [nn.LocalResponseNorm(size=5 ,alpha=0.0001, beta=0.75, k=2)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [(96, 7, 2, 1), ('M', 0, 0, 0), (256, 5, 2, 0), ('M', 0, 0, 0),
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
            center_frame = [random.randint(5, num_of_frame - 6)]  # num of the chosen frame
            myTransforms = transforms.Compose([
                transforms.RandomCrop(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            center_frame = range(5, num_of_frame - 6, int((num_of_frame-11) / self.T))[:self.T]  # num of the chosen frame
            myTransforms = transforms.Compose([
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
            ])

        # batch_flag = 1
        stack_center = []
        for center in center_frame:

            chosen_frames = range(center - 4, center + 6)
            # flag = 1
            stack_frame = []

            for i in chosen_frames:
                frame_path = 'frame%06d.jpg' % i

                for dirc in dircs:

                    file_path = os.path.join(self.image_dir, dirc)
                    file_path = os.path.join(file_path, name_of_file)
                    file_path = os.path.join(file_path, frame_path)

                    img = Image.open(file_path)

                    stack_frame.append(myTransforms(img))

                    # frame = myTransforms(img).numpy()
                    #
                    # if flag == 1:
                    #     train_data = frame.copy()
                    #     flag = 0
                    # else:
                    #     train_data = np.append(train_data, frame, axis=0)

            train_data = torch.stack(stack_frame, dim=0)
            stack_center.append(train_data)

            # train_data = np.expand_dims(train_data, axis=0)
            # if batch_flag == 1:
            #     minibatch_data = train_data.copy()
            #     batch_flag = 0
            # else:
            #     minibatch_data = np.append(minibatch_data, train_data, axis=0)

        # inputs = torch.Tensor(minibatch_data)
        inputs = torch.stack(stack_center, dim=0)
        label = self.label_list[index]
        return inputs.squeeze(), label

    def __len__(self):
        return len(self.file_list)

# Define a new evaluation function
def evaluate():
    with torch.no_grad():
        net.eval()

        running_error = 0
        num_batches = 0

        for minibatch_data, minibatch_label in test_data:

            B, T, C, H, W = list(minibatch_data.size())
            inputs = minibatch_data.view(B * T, C, H, W)
            inputs = inputs.to(device)
            minibatch_label = minibatch_label.to(device)

            scores = net(inputs)

            scores = scores.view(B, T, -1)
            scores = torch.mean(scores, 1)
            predict = scores.argmax(dim=1)

            error = get_error(scores, minibatch_label)

            running_error += error.item()

            num_batches += 1

            # release caches
            del minibatch_data, scores

            # if num_batches % 10 == 0:
            #     print('epochs = ', num_batches, 'error = ', (running_error/num_batches)*100, 'percent')

        total_error = running_error/num_batches
        print('time =', (time.time()-start)/60, '\t error on test set =',  total_error * 100, 'percent')

        writer.add_scalar('test/error', total_error, iterations)
        writer.add_scalar('test/accuracy', 1-total_error, iterations)

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
net = torchvision.models.vgg16_bn(pretrained=False)
net.features._modules['0'] = nn.Conv2d(20, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
net.classifier._modules['2'] = nn.Dropout(p=0.9, inplace=False)
net.classifier._modules['5'] = nn.Dropout(p=0.9, inplace=False)
net.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 101))

# net = torchvision.models.vgg16()
# net.load_state_dict(torch.load("t_params_about50k.pkl"))

# net = rgb_vgg16()

net = net.to(device)
# print(net)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)

# Initialize learning rate
my_lr = 0.01
bs = 64
accumulation_steps = 2

# Load the data
train_dataset = myDataset(train_label, trainlist, train_frame_count, path, 0)
train_data = DataLoader(train_dataset,
                        batch_size=bs,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True
                        )
test_dataset = myDataset(test_label, testlist, test_frame_count, path, 3)
test_data = DataLoader(test_dataset,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        )

# set up the writer
writer = SummaryWriter()

#=============== Training =================
print("Start training")
start = time.time()
iterations = 0
evaluate()
for epoch in range(1, int(2200*256/bs)):

    net.train()

    # set running quatities
    running_loss = 0
    running_error = 0
    num_batches = 0

    # create an optimizer
    # FOR TRAINING FROM SCRATCH
    # update the learning rate w.r.t. iterations num
    if iterations >= 50000*4 and iterations < 70000*4:
        my_lr = 0.001
        optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)
    elif iterations >= 70000*4 and iterations < 80000*4:
        my_lr = 0.0001
        optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)
    elif iterations >= 80000*4:
        print("End of training")
        break
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)

    # FOR FINE-TUNING
    # if iterations >= 14000 and iterations < 20000:
    #     my_lr = 0.001
    #     optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)
    # elif iterations >= 20000:
    #     break
    # else:
    #     optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)

    # optimizer = torch.optim.Adam(net.parameters())
    # net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    optimizer.zero_grad()

    # training in one epoch
    for minibatch_data, minibatch_label in train_data:

        # send them to the device
        inputs = minibatch_data.squeeze()
        inputs = inputs.cuda(non_blocking=True)
        minibatch_label = minibatch_label.cuda(non_blocking=True)

        # start tracking all operations
        inputs.requires_grad_()

        # forward inputs through the net
        scores = net(inputs)

        # compute the loss
        loss = criterion(scores, minibatch_label)

        #============APEX============
        # optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        # optimizer.step()
        # iterations += 1

        # ===========TRICK===========
        # # trick ENABLED
        # loss = loss / accumulation_steps
        # loss.backward()
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

        # # check point ENABLED
        # # set the gradients to zeros
        # optimizer.zero_grad()
        # num_segments = 2
        # x = checkpoint_sequential(net, num_segments, inputs)
        # # backward pass
        # loss.backward()
        # # one step of SGD
        # optimizer.step()
        # iterations += 1

        # ================END=================

        # compute stats
        # add the loss to the running loss
        running_loss += loss.detach().item()

        # compute the error
        error = get_error(scores.detach(), minibatch_label)
        running_error += error.item()
        num_batches += 1

        # clear cache
        del inputs, scores, loss

    # stats of the full training set
    total_loss = running_loss/num_batches
    total_error= running_error/num_batches
    clock = (time.time()-start)/60
    print('epoch=', epoch, '\t time=', clock, '\t lr=', my_lr, '\t loss=', running_loss,
          '\t error=', total_error * 100, '\t iterations=', iterations)

    if epoch % 10 == 0:
        torch.save(net.state_dict(), 't_vgg_scra.pkl')
        evaluate()

    # ==================== tensorboard=====================#
    writer.add_scalar('train/loss', total_loss, iterations)
    writer.add_scalar('train/error', total_error, iterations)
    writer.add_scalar('train/accuracy', 1 - total_error, iterations)
