import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from network import CNN
from utils import graph, losses_to_ewma
from data_feeder import DataFeeder

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


def evaluate_acc(cnn, mnist_test, i):
    print(i)
    
    cnn.model.eval()

    images, labels = mnist_test
    out = cnn.model(images)
    _, index = out.topk(1, dim=1)
    correct = torch.mean((index.squeeze() == labels).float())

    cnn.acc += [correct.data.cpu().numpy()[0]]
    cnn.acc_indices += [i]

    loss = F.cross_entropy(out, labels)
    cnn.val_losses += [loss.data.cpu().numpy()[0]]
    cnn.val_x_indices += [i]


    cnn.model.train()
    
        
def train(model_info, images, labels, i):
    model_info.optim.zero_grad()
    pred = model_info.model(images)
    loss = F.cross_entropy(pred, labels)
    loss.backward()
    model_info.optim.step()

    model_info.losses += [loss.data.cpu().numpy()[0]]
    model_info.x_indices += [i]

    
def main():
    batch_size = 100
    mnist_train = datasets.MNIST("/hdd/Data/MNIST/", train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = datasets.MNIST("/hdd/Data/MNIST/", train=False, transform=transforms.ToTensor(), download=True)
    data_feeder = DataFeeder(mnist_train, preprocess_workers = 1, cuda_workers = 1, cpu_size = 12,
                 cuda_size = 3, batch_size = batch_size, use_cuda = True, volatile = False)
    data_feeder.start_queue_threads()
    cnn = ModelStruct(CNN().cuda(), 0.001)

    test_data = make_batch(len(mnist_test), 0, mnist_test, use_cuda = True)

    #300k
    #175k
    for i in range(100001):
        images, labels = data_feeder.get_batch()
        train(cnn, images, labels, i)
        if i % 100 == 0:
            evaluate_acc(cnn, test_data, i)
        # decrease learning rate
        if i in [300000]:
            cnn.lr = cnn.lr/10
            print("updated learning rate: current lr:", cnn.lr)
            for param_group in cnn.optim.param_groups:
                param_group['lr'] = cnn.lr
        if i % 100000 == 0:
            torch.save(cnn.model, "savedir/model_it"+str(i//1000)+"k.pt")
            
    print(max(cnn.acc))
    print(cnn.acc)
    graph(cnn)
    cnn.losses = losses_to_ewma(cnn.losses)
    cnn.val_losses = losses_to_ewma(cnn.val_losses)
    cnn.acc = losses_to_ewma(cnn.acc)
    graph(cnn)

    
if __name__ == "__main__":
    main()
