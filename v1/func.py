import torch
import torchvision
import time

def getData(batch_size, transform):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return(trainloader, testloader, classes)



def evaluate(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def train(model, trainloader, testloader, criterion, optimizer, device, epochs):
    model.to(device)
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_acc = evaluate(model, trainloader, device)
        test_acc = evaluate(model, testloader, device)

        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, epochs, running_loss))
        print('Training Accuracy: %.2f %%' % (train_acc))
        print('Test Accuracy: %.2f %%' % (test_acc))

    end_time = time.time()
    total_time = end_time - start_time
    total_flops = (2 * 3 * 5 * 5 * 6 * 14 * 14 + 6 * 5 * 5 * 16 * 5 * 5 + 16 * 5 * 5 * 120 + 120 * 84 + 84 * 10) * len(trainloader.dataset)
    total_adds = total_flops // 2
    total_muls = total_flops // 2
    total_maxs = (2 * 5 * 5 * 6 * 14 * 14 + 2 * 5 * 5 * 16 * 5 * 5) * len(trainloader.dataset)
    total_ops = total_adds + total_muls + total_maxs
    ops_per_second = total_ops / total_time

    print('Total FLOPs: %.2f' % (total_flops))
    print('Total Additions: %.2f' % (total_adds))
    print('Total Multiplications: %.2f' % (total_muls))
    print('Total Maximums: %.2f' % (total_maxs))
    print('Total Operations: %.2f' % (total_ops))
    print('Total Time: %.2f seconds' % (total_time))
    print('Operations Per Second: %.2f' % (ops_per_second))
    print('Finished Training')
