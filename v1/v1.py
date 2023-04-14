from model import Net
import func
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

def main():
    ######################################################################
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4

    trainloader, testloader, classes = func.getData(batch_size, transform)

    ######################################################################

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    ######################################################################

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################################################################

    func.train(net, trainloader, testloader, criterion, optimizer, device, epochs=2)

    ######################################################################

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    main()