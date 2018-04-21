import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from network import FCC, CNN
from data_feeder import DataFeeder
from logger import Logger

import random

def create_train_val_test_split(batch_size):
    mnist_train = datasets.MNIST("/hdd/Data/MNIST/", train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = datasets.MNIST("/hdd/Data/MNIST/", train=False, transform=transforms.ToTensor(), download=True)

    indices = random.sample(range(60000), 5000)
    train_data = []
    val_data = []
    test_data = []
    for i, ex in enumerate(mnist_train):
        if i in indices:
            val_data += [ex]
        else:
            train_data += [ex]
    for ex in mnist_test:
        test_data += [ex]

    return train_data, val_data, test_data

        
def make_batch(batch_size, i, data, use_cuda = True, volatile = True):
    images = []
    labels = []
    for j in range(batch_size):
        image, label = data[(i+j) % len(data)]
        images += [image]
        labels += [label]
    images_tensor = torch.stack(images).resize_(batch_size, 1, 28*28)
    labels_tensor = torch.LongTensor(labels)
    if use_cuda:
        images_tensor = images_tensor.cuda()
        labels_tensor = labels_tensor.cuda()
    return Variable(images_tensor, volatile=volatile), Variable(labels_tensor, volatile=volatile)


def evaluate_acc(batch_size, model, data, i, val_loss, val_acc, permute):
    model.eval()

    images, labels = data
    accs = []
    losses = []
    for j in range(0, len(images), batch_size):
        #loss
        batch_images = images[j:j+batch_size]
        batch_labels = labels[j:j+batch_size]
        
        pred = model(batch_images)
        losses += [F.cross_entropy(pred, batch_labels)]

        #acc
        _, index = pred.topk(1, dim=1)
        accs += [torch.mean((index.squeeze() == batch_labels).float())]
        
    acc = torch.mean(torch.stack(accs))
    print(acc)
    loss = torch.mean(torch.stack(losses))
    val_loss.write_log(loss, i)
    val_acc.write_log(acc, i)

    model.train()    


def main(permute):
    batch_size = 100
    train_data, val_data, test_data = create_train_val_test_split(batch_size)
    val_data = make_batch(len(val_data), 0, val_data, use_cuda = True, volatile = True)
    test_data = make_batch(len(test_data), 0, test_data, use_cuda = True, volatile = True)

    cnn = CNN().cuda()
    fcc = FCC().cuda()
    
    cnn_test_loss = Logger("cnn_test_losses.txt")
    cnn_test_acc = Logger("cnn_test_acc.txt")
    fcc_test_loss = Logger("fcc_test_losses.txt")
    fcc_test_acc = Logger("fcc_test_acc.txt")
    for i in range(0, 100001, 1000):
        print(i)
        cnn.load_state_dict(torch.load("savedir/cnn_it"+str(i//1000)+"k.pth"))
        evaluate_acc(batch_size, cnn, test_data, i, cnn_test_loss, cnn_test_acc, permute)
        fcc.load_state_dict(torch.load("savedir/fcc_it"+str(i//1000)+"k.pth"))
        evaluate_acc(batch_size, fcc, test_data, i, fcc_test_loss, fcc_test_acc, permute)

    
if __name__ == "__main__":
    main(None)
