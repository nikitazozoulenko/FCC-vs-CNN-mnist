import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

from network import CNN, FCC
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
    
        
def train(model, optimizer, images, labels, i, train_logger, permute):
    optimizer.zero_grad()
    # only take the loss of the last time step
    pred = model(images)
    loss = F.cross_entropy(pred, labels)
    loss.backward()
    optimizer.step()

    train_logger.write_log(loss, i)


def increase_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("updated learning rate: new lr:", param_group['lr']*10)
        param_group['lr'] = param_group['lr']*10


def decrease_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("updated learning rate: new lr:", param_group['lr']/10)
        param_group['lr'] = param_group['lr']/10
    

def main():
    batch_size = 100
    train_data, val_data, test_data = create_train_val_test_split(batch_size)
    data_feeder = DataFeeder(train_data, preprocess_workers = 1, cuda_workers = 1, cpu_size = 10,
                 cuda_size = 10, batch_size = batch_size, use_cuda = True, volatile = False)
    data_feeder.start_queue_threads()
    val_data = make_batch(len(val_data), 0, val_data, use_cuda = True, volatile = True)
    test_data = make_batch(len(test_data), 0, test_data, use_cuda = True, volatile = True)

    cnn = CNN().cuda()
    fcc = FCC().cuda()

    optimizer_cnn = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
    optimizer_fcc = optim.SGD(fcc.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)

    cnn_train_loss = Logger("cnn_train_losses.txt")
    cnn_val_loss = Logger("cnn_val_losses.txt")
    cnn_val_acc = Logger("cnn_val_acc.txt")
    fcc_train_loss = Logger("fcc_train_losses.txt")
    fcc_val_loss = Logger("fcc_val_losses.txt")
    fcc_val_acc = Logger("fcc_val_acc.txt")

    #permute = Variable(torch.from_numpy(np.random.permutation(28*28)).long().cuda(), requires_grad=False)
    permute = None

    for i in range(100001):
        images, labels = data_feeder.get_batch()
        train(cnn, optimizer_cnn, images, labels, i, cnn_train_loss, permute)
        train(fcc, optimizer_fcc, images, labels, i, fcc_train_loss, permute)
        if i % 100 == 0:
            print(i)
            evaluate_acc(batch_size, cnn, val_data, i, cnn_val_loss, cnn_val_acc, permute)
            evaluate_acc(batch_size, fcc, val_data, i, fcc_val_loss, fcc_val_acc, permute)
        if i in [70000, 90000]:
            decrease_lr(optimizer_cnn)
            decrease_lr(optimizer_fcc)
        if i % 1000 == 0:
            torch.save(cnn.state_dict(), "savedir/cnn_it"+str(i//1000)+"k.pth")
            torch.save(fcc.state_dict(), "savedir/fcc_it"+str(i//1000)+"k.pth")
            
    data_feeder.kill_queue_threads()

    import evaluate
    evaluate.main(permute)

    
if __name__ == "__main__":
    main()
