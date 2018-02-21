import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from network import CNN
from utils import graph, losses_to_ewma

class ModelStruct(object):
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        # self.optim = optim.Adam(self.model.parameters(), lr=lr)
        self.losses = []
        self.x_indices = []
        self.val_losses = []
        self.val_x_indices = []
        self.acc = []
        self.acc_indices = []

        
def make_batch(batch_size, i, data, use_cuda = True):
    images = []
    labels = []
    for j in range(batch_size):
        image, label = data[(i+j) % len(data)]
        images += [image]
        labels += [label]
    images_tensor = torch.stack(images)
    labels_tensor = torch.LongTensor(labels)
    if use_cuda:
        images_tensor = images_tensor.cuda()
        labels_tensor = labels_tensor.cuda()
    return Variable(images_tensor), Variable(labels_tensor)


def evaluate_acc(cnn, mnist_val, i, batch_size, use_cuda = True):
    print(i)
    
    cnn.model.eval()

    cnn_correct = 0

    for j in range(len(mnist_val)//batch_size):
        images, labels = make_batch(batch_size, j, mnist_val, use_cuda)
        cnn_correct += eval_single(cnn, images, labels) / len(mnist_val)

    cnn.acc += [cnn_correct.data.cpu().numpy()[0]]
    cnn.acc_indices += [i]

    cnn.model.train()

        
def eval_single(model_info, images, labels):
    out = model_info.model(images)
    _, index = out.topk(1, dim=1)
    correct = torch.sum(index.squeeze() == labels).float()
    return correct

        
def train(model_info, images, labels, val_images, val_labels, i, batch_size):
    model_info.optim.zero_grad()
    pred = model_info.model(images)
    loss = F.cross_entropy(pred, labels)
    loss.backward()
    model_info.optim.step()
    
    val_pred = model_info.model(val_images)
    val_loss = F.cross_entropy(val_pred, val_labels)

    model_info.losses += [loss.data.cpu().numpy()[0]]
    model_info.x_indices += [i]
    model_info.val_losses += [val_loss.data.cpu().numpy()[0]]
    model_info.val_x_indices += [i]

    
def main():
    to_PIL = transforms.ToPILImage()
    mnist_train = datasets.MNIST("/hdd/Data/MNIST/", train=True, transform=transforms.ToTensor(), download=True)
    mnist_val = datasets.MNIST("/hdd/Data/MNIST/", train=False, transform=transforms.ToTensor(), download=True)

    cnn = ModelStruct(CNN().cuda(), 0.001)

    #300k
    #175k
    batch_size = 100
    for i in range(30001):
        images, labels = make_batch(batch_size, i, mnist_train, use_cuda = True)
        val_images, val_labels = make_batch(batch_size, i, mnist_val, use_cuda = True)
        train(cnn, images, labels, val_images, val_labels, i, batch_size)
        if i % 100 == 0:
            evaluate_acc(cnn, mnist_val, i, batch_size, use_cuda = True)
    # decrease learning rate
    if i in [15000]:
        cnn.lr = cnn.lr/10
        print("updated learning rate: current lr:", cnn.lr)
        for param_group in cnn.optim.param_groups:
            param_group['lr'] = learning_rate
    if i % 100000000000000 == 0 and i != 0:
        torch.save(model, "savedir/model"+"it"+str(i//1000)+"k.pt")
            

    print(max(cnn.acc))
    graph(cnn)

    cnn.losses = losses_to_ewma(cnn.losses)
    cnn.val_losses = losses_to_ewma(cnn.val_losses)
    cnn.acc = losses_to_ewma(cnn.acc)

    graph(cnn)

    
if __name__ == "__main__":
    main()
