import os
import pandas as pd

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
    trainlist[i] = trainlist[i].split(' ',1)[0].split('/',1)[1].split('.',1)[0]
print("Train data loaded")

for i in range(len(testlist)):
    label = testlist[i].split('/',1)[0]
    test_label.append(label2index[label])
    testlist[i] = testlist[i].split('/',1)[1].split('.',1)[0]
print("Test  data loaded")

# count number of frames for each video
train_frame_count = []    # num of frames of each video
test_frame_count = []
path = 'ucf101_jpegs_256/jpegs_256' # __ training data in total

for line in trainlist:
    # count the num of frames in each sub-directory
    sub_path = os.path.join(path, line)
    train_frame_count.append(0)
    for item in os.listdir(sub_path):
        train_frame_count[-1] += 1

for line in testlist:
    # count the num of frames in each sub-directory
    sub_path = os.path.join(path, line)
    test_frame_count.append(0)
    for item in os.listdir(sub_path):
        test_frame_count[-1] += 1
print("Frames number counted")

df = pd.DataFrame(train_frame_count)
df.to_csv('train_frame_count.csv', header=None, index=False)
df = pd.DataFrame(test_frame_count)
df.to_csv('test_frame_count.csv', header=None, index=False)
