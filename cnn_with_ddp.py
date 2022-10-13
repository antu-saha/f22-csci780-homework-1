import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from model import ConvNet
from utils import load_dataset, load_checkpoint, train, test


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

    """def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)"""

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
                    # writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
                    # writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
                    running_loss = 0.0
                    running_correct = 0

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self.save_checkpoint(epoch)

        print('Finished Training...')


        """for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)"""


"""def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = ConvNet  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer"""


"""def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )"""


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # hyper parameters
    # save_every = 5
    # total_epochs = 10
    # batch_size = 1000
    learning_rate = 0.001

    ddp_setup(rank, world_size)
    train_data, test_data, classes = load_dataset(batch_size)
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # dataset, model, optimizer = load_train_objs()
    # train_data = prepare_dataloader(train_loader, batch_size)

    trainer = Trainer(model, train_data, optimizer, criterion, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

    test(device, batch_size, test_data, model, classes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1000, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.total_epochs, args.save_every, args.batch_size), nprocs=world_size)
