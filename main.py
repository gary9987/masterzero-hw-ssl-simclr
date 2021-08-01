import numpy as np
from simclrModel import SimCLR
from simclr.modules import NT_Xent
from torch.utils.data import DataLoader
import torchvision
from transform import TransformsSimCLR
import torch
from tqdm import tqdm
from simclr.modules.resnet_hacks import modify_resnet_model


def train(train_loader, model, criterion, optimizer):
    model.train()
    loss_epoch = 0

    for (x_i, x_j), _ in train_loader:
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch

def test(test_loader, model, criterion):
    model.eval()
    loss_epoch = 0

    for (x_i, x_j), _ in test_loader:
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)

        loss_epoch += loss.item()

    return loss_epoch

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = torchvision.datasets.CIFAR10(
        './data',
        download=True,
        train=True,
        transform=TransformsSimCLR(size=32)
    )

    batch_size = 512

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        sampler=None,
        pin_memory=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        './data',
        download=True,
        train=False,
        transform=TransformsSimCLR(size=32).test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        sampler=None,
        pin_memory=True
    )

    # initialize ResNet
    encoder = modify_resnet_model(torchvision.models.resnet18())
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, hidden_dim=512, projection_dim=128, n_features=n_features)
    model = model.to(device)

    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # world_size might mean amount of gpus
    criterion = NT_Xent(batch_size, temperature=0.5, world_size=1)

    n_epochs = 200

    glob_loss = np.Inf
    for epoch in range(1, n_epochs + 1):

        train_loss = train(train_loader, model, criterion, optimizer) / len(train_loader)
        test_loss = test(test_loader, model, criterion) / len(test_loader)

        print(f"Epoch [{epoch}/{n_epochs}]\t Train_Loss: {train_loss}\t Test_Loss: {test_loss}")

        if test_loss < glob_loss:
            print('Test loss decreased ({:.6f} --> {:.6f}).  Saving checkpoint ...'.format(glob_loss, test_loss))
            glob_loss = test_loss
            torch.save(model.state_dict(), 'checkpoint1.pth')
