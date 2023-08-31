from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import main


def MNIST(args):
    batch_size = args.batch_size

    if args.model == 'LeNet':
        torchvision_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    train_dataset = datasets.MNIST(root='../../PytorchZeroToAll/data/',
                                   train=True,
                                   transform=torchvision_transform,
                                   download=True)
    test_dataset = datasets.MNIST(root='../../PytorchZeroToAll/data/',
                                  train=False,
                                  transform=torchvision_transform)

    print("Generate TrainLoader")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)
    return train_loader, test_loader
