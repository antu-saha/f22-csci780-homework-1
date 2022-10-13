import torch
import torch.nn as nn
from model import ConvNet
import torchvision.models as models
from utils import load_dataset, load_checkpoint, train, test
from torch.utils.tensorboard import SummaryWriter
import wandb

if __name__ == '__main__':
    wandb.init(project="f22-csci780-homework-1")
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 4,
        "batch_size": 200
    }

    writer = SummaryWriter("runs/cifar10")

    ckp_path = 'checkpoints/checkpoint.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # hyper parameters
    num_epochs = 100
    batch_size = 200
    learning_rate = 0.001

    print(f'Please select one from the following menu:')
    print(f'For CNN, press: 1')
    print(f'For ResNet-18, press: 2')

    user_input = input('Enter your choice:')
    user_input = int(user_input)

    if user_input == 1:
        model = ConvNet()
    elif user_input == 2:
        model = models.resnet18(pretrained=True)

    train_loader, test_loader, classes = load_dataset(batch_size)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # model, optimizer, epoch, loss = load_checkpoint(ckp_path, model, optimizer)
    model.train()
    train(writer, ckp_path, train_loader, num_epochs, device, model, criterion, optimizer)

    model.eval()
    test(device, batch_size, test_loader, model, classes)
