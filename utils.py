import torch
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import accuracy, precision, recall
from torchmetrics.functional import precision_recall
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_dataset(batch_size):
    # dataset has PILImage images of range [0, 1].
    # we transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root='./CIFAR_Dataset', train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./CIFAR_Dataset', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    return train_loader, test_loader, classes


def load_dataset_for_ddp(batch_size):
    # dataset has PILImage images of range [0, 1].
    # we transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root='./CIFAR_Dataset', train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./CIFAR_Dataset', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    return train_loader, test_loader, classes


def save_checkpoint(ckp_path, model, epoch, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, ckp_path)


def load_checkpoint(ckp_path, model, optimizer):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train(writer, ckp_path, train_loader, num_epochs, device, model, criterion, optimizer):
    n_total_steps = len(train_loader)
    running_loss = 0.0
    running_correct = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}, Step [{i + 1}/{n_total_steps}]'
                      f'Loss: {loss.item():.4f}]')
                writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
                writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
                wandb.log({"Epoch": epoch + 1, "Loss": loss, "Model Accuracy:": running_correct / 100})
                running_loss = 0.0
                running_correct = 0



            # if epoch % 1 == 0:
    save_checkpoint(ckp_path, model, epoch, optimizer, loss)

    print('Finished Training...')


def test(device, batch_size, test_loader, model, classes):
    with torch.no_grad():
        target = torch.tensor([])
        target = target.type(torch.LongTensor)
        target = target.to(device)
        prediction = torch.tensor([])
        prediction = prediction.type(torch.LongTensor)
        prediction = prediction.to(device)

        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            target = torch.cat((target, labels), 0)
            prediction = torch.cat((prediction, predicted), 0)

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        print(f'# of Samples: {n_samples}')
        print(target)
        print(len(target))
        print(prediction)
        print(len(prediction))
        average = 'macro'
        print('Metrics:')
        # print(f'ACC: {accuracy_score(target, prediction)}')
        # print(f'REC: {recall_score(target, prediction, average=average)}')
        # print(f'PREC: {precision_score(target, prediction, average=average)}')
        net_accuracy = accuracy(prediction, target, num_classes=10)
        print(f'Accuracy of the network: {net_accuracy}')
        net_recall = recall(prediction, target, average=average, num_classes=10)
        print(f'Recall of the network: {net_recall}')
        net_precision = precision(prediction, target, average=average, num_classes=10)
        print(f'Precision of the network: {net_precision}')
        net_f1_score = (2 * net_precision * net_recall) / (net_precision + net_recall)
        print(f'F1 score of the network: {net_f1_score}')

        # print(f'TM P_R: {precision_recall(prediction, target, average=average, num_classes=10)}')
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network (manual calculation): {acc} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]} (manual calculation): {acc} %')
