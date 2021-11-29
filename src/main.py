import numpy as np
import torch
import random
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
from torch import nn, optim
from torch.nn import functional as F
import pickle
import argparse
import sys
import os
os.environ['display'] = 'localhost:14.0'

import PIL
import time
from tqdm import tqdm
import gc

from myModels import VGG16
from myDatasets import Digit_detection, CIFAR10, CIFAR100

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--cuda', type = int, default = '-1')
parser.add_argument('--dataset', type = str, default = 'CIFAR10', choices = ["CIFAR10", "CIFAR100", "DIGITS"])
args = parser.parse_args()

seed_torch(args.seed)

if args.cuda == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 20
lr = 1e-3
SAVE_INTERVAL = 1
NUM_EPOCHS = 100
torch.autograd.set_detect_anomaly(True)

checkpoint_path = './checkpoints/{}/'.format(args.dataset)
if not os.path.isdir(checkpoint_path):
    os.system('mkdir -p {}'.format(checkpoint_path))
if not os.path.isdir('../results/{}/train'.format(args.dataset)):
    os.system('mkdir -p ../results/{}/train'.format(args.dataset))
if not os.path.isdir('../results/{}/test'.format(args.dataset)):
    os.system('mkdir -p ../results/{}/test'.format(args.dataset))

# train_transform = transforms.Compose([transforms.Resize((224,224))])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])
# test_transform = None

if args.dataset == 'CIFAR10':
    DataObject = CIFAR10
    NUM_CLASSES = 10
elif args.dataset == 'CIFAR100':
    DataObject = CIFAR100
    NUM_CLASSES = 100
elif args.dataset == 'DIGITS':
    DataObject = Digit_detection
    NUM_CLASSES = 2

trainset = DataObject(train = 'train', transform=train_transform, bootstrapping = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle = True)
testset = DataObject(train = 'test', transform=test_transform, bootstrapping = True)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle = False)
model = VGG16(n_classes=NUM_CLASSES, pretrained = True, input_channels = 3).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.5, 0.999))
criterion = nn.CrossEntropyLoss().to(device)

if args.resume:
    model_state = torch.load(checkpoint_path + 'state.pth', map_location = device)['state']
    if (not args.state == -1):
        model_state = args.state
    print('Loading checkpoint at model state {}'.format(model_state))
    dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state), map_location = device)
    pre_e = dic['e']
    model.load_state_dict(dic['model'])
    optimizer.load_state_dict(dic['optimizer'])
    losses = dic['losses']
    accuracies = dic['accuracies']
    # testaccuracies = dic['testaccuracies']
    print('Resuming Training after {} epochs'.format(pre_e))
else:
    model_state = 0
    pre_e =0
    losses = []
    accuracies = []
    # testaccuracies = []
    print('Starting Training')

def train(e):
    print('\nTraining for epoch {}'.format(e))
    tot_loss = 0
    tot_correct = 0
    tot = 0

    for batch_num,(images, labels) in tqdm(enumerate(trainloader), desc = 'Epoch {}'.format(e), total = len(trainloader)):
        optimizer.zero_grad()
        model.train()
        outp = model(images.to(device))
        loss = criterion(outp, labels.to(device))
        loss.backward()
        optimizer.step()

        preds = torch.max(outp,1)[1]
        tot_correct += (preds == labels.to(device)).sum()
        tot += images.shape[0]
        tot_loss += loss.item()

    trainset.generate_index_mapping()
    print('Total Loss for epoch = {}'.format(tot_loss/batch_num))
    print('Total Train Accuracy for epoch = {}'.format(100*tot_correct/tot))
    return tot_loss/batch_num, 100*tot_correct/tot

def test(e):
    with torch.no_grad():
        print('\nTesting after {} epochs'.format(e))
        tot_correct = 0
        tot = 0

        for batch_num,(images, labels) in tqdm(enumerate(testloader), desc = 'Epoch {}'.format(e), total = len(testloader)):
            model.eval()
            outp = model(images.to(device))
            preds = torch.max(outp,1)[1]
            tot_correct += (preds == labels.to(device)).sum()
            tot += images.shape[0]

        print('Total Test Accuracy for epoch = {}\n'.format(100*tot_correct/tot))
        return 100*tot_correct/tot

def visualize(train = False):
    if train:
        str_k = 'train'
        dloader = trainloader
    else:
        str_k = 'test'
        dloader = testloader
    
    if args.dataset == 'CIFAR10':
        num_vis = 5
    elif args.dataset == 'CIFAR100':
        DataObject = CIFAR100
        num_vis = 2
    elif args.dataset == 'DIGITS':
        num_vis = 10
    with torch.no_grad():
        stored_ims = torch.zeros(NUM_CLASSES, num_vis, 3, 224,224)
        counter = torch.zeros(NUM_CLASSES).int()
        for batch_num,(images, labels) in tqdm(enumerate(dloader), desc = 'Visualising {} images'.format(str_k), total = len(dloader)):
            for ci in range(NUM_CLASSES):
                if counter[ci] == num_vis:
                    continue
                indices = labels == ci
                tnum = min(indices.sum(), num_vis-counter[ci])
                stored_ims[ci, counter[ci]:counter[ci]+tnum,:,:,:] = images[indices][:tnum,:,:,:]
                counter[ci] += tnum
            if counter.sum() == num_vis*NUM_CLASSES:
                break

    for ci in range(NUM_CLASSES):
        saliencies = model.get_saliency_maps(stored_ims[ci].to(device), ci).cpu()
        cams = model.get_CAM(stored_ims[ci].to(device), ci).cpu()

        for i in range(cams.shape[0]):
            im = stored_ims[ci,i]
            cam = cams[i]
            sal = saliencies[i]
            
            fig = plt.figure(figsize = (9,3))
            plt.subplot(1,3,1)
            plt.imshow(torch.permute(im, (1,2,0)))
            plt.title('Data Image')
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(cam, cmap = 'jet')
            plt.title('CAM')
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(sal, cmap = 'jet')
            plt.title('Saliency Map')
            plt.axis('off')
            plt.savefig('../results/{}/{}/class_{}_{}.png'.format(args.dataset, str_k, ci,i))
            plt.close('all')

if args.eval:
    visualize(train = True)
    visualize(train = False)
    test(pre_e)
    with open('status_{}.txt'.format(args.dataset), 'w') as f:
        f.write('1')
    os._exit(0)

for e in range(NUM_EPOCHS):
    if pre_e > 0:
        pre_e -= 1
        continue

    l, a = train(e)
    # visualize(train = True)
    # visualize(train = False)
    # ta = test(e)
    losses.append(l)
    accuracies.append(a)
    # testaccuracies.append(ta)
    
    dic = {}
    dic['e'] = e+1
    dic['model'] = model.state_dict()
    dic['optimizer'] = optimizer.state_dict()
    dic['losses'] = losses
    dic['accuracies'] = accuracies
    # dic['testaccuracies'] = testaccuracies


    if (e+1) % SAVE_INTERVAL == 0:
        torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
        torch.save({'state': model_state}, checkpoint_path + 'state.pth')
        print('Saving model after {} Epochs'.format(e+1))
