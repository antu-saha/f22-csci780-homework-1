import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from model import ConvNet
from utils import load_dataset_for_ddp, test
import wandb


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)  # gloo not nccl


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.CrossEntropyLoss,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        n_total_steps = len(self.train_data)
        running_loss = 0.0
        running_correct = 0
        for epoch in range(max_epochs):
            for i, (images, labels) in enumerate(self.train_data):
                # origin shape: [4, 3, 32, 32] = 4, 3, 1024
                # input_layer: 3 input channels, 6 output channels, 5 kernel size
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)

                # forward pass
                outputs = self.model(images).to(self.gpu_id)
                loss = self.criterion(outputs, labels).to(self.gpu_id)

                # backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_correct += (predicted == labels).sum().item()

                if (i + 1) % 50 == 0:
                    print(f'Epoch [{epoch + 1}/{max_epochs}, Step [{i + 1}/{n_total_steps}]'
                          f'Loss: {loss.item():.4f}]')
                    wandb.log({"Epoch": epoch + 1, "Loss": loss, "Model Accuracy:": running_correct / 100})

                    running_loss = 0.0
                    running_correct = 0

            if self.gpu_id == 0 and (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1)

        print('Finished Training...')


def main(rank: int, world_size: int, total_epochs: int, save_every: int, batch_size: int, learning_rate: float):
    ddp_setup(rank, world_size)
    train_data, test_data, classes = load_dataset_for_ddp(batch_size)
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, train_data, optimizer, criterion, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    # hyper parameters
    save_every = 5
    num_epochs = 10
    batch_size = 1000
    learning_rate = 0.001
    world_size = torch.cuda.device_count()

    wandb.init(project="f22-csci780-homework-1")
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }

    mp.spawn(main, args=(world_size, num_epochs, save_every, batch_size, learning_rate), nprocs=world_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = ConvNet()
    model = torch.load("checkpoint.pt")
    model = model.to(device)
    train_data, test_data, classes = load_dataset_for_ddp(batch_size)
    test(device, batch_size, test_data, model, classes)
