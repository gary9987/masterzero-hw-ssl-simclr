from simclrModel import SimCLR
from simclr.modules import LogisticRegression
from torch.utils.data import DataLoader
import torchvision
from simclr.modules.transformations import TransformsSimCLR
import torch
from tqdm import tqdm
import numpy as np
from simclr.modules.resnet_hacks import modify_resnet_model
import torch.utils.data as data

import nni

def train(loader, device, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.train()

    for x, y in tqdm(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def test(loader, device, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()

    for x, y in tqdm(loader):
        model.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []

    for x, y in tqdm(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(context_model, train_loader, valid_loader, test_loader, device):
    train_X, train_y = inference(train_loader, context_model, device)
    valid_X, valid_y = inference(valid_loader, context_model, device)
    test_X, test_y = inference(test_loader, context_model, device)
    return train_X, train_y, valid_X, valid_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    valid = torch.utils.data.TensorDataset(
        torch.from_numpy(X_valid), torch.from_numpy(y_valid)
    )
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, valid_loader, test_loader


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = torchvision.datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=TransformsSimCLR(size=32).test_transform
    )

    # Random split training dataset to training set and validation set
    train_set_size = int(len(train_dataset) * args['training_size'])
    valid_set_size = len(train_dataset) - train_set_size
    train_dataset, valid_dataset = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(42))

    test_dataset = torchvision.datasets.CIFAR10(
        './data',
        train=False,
        download=True,
        transform=TransformsSimCLR(size=32).test_transform
    )

    batch_size = args['batch_size']

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )

    encoder = modify_resnet_model(torchvision.models.resnet18())
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, hidden_dim=512, projection_dim=128, n_features=n_features)
    model_fp = 'checkpoint1.pth'
    simclr_model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
    simclr_model = simclr_model.to(device)
    simclr_model.eval()

    ## Logistic Regression (classifier)
    n_classes = 10  # stl-10 / cifar-10
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, valid_X, valid_y, test_X, test_y) = get_features(
        simclr_model, train_loader, valid_loader, test_loader, device
    )

    arr_train_loader, arr_valid_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, valid_X, valid_y, test_X, test_y, batch_size
    )

    n_epochs = 130
    glob_loss = np.Inf
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss_epoch, train_accuracy_epoch = train(arr_train_loader, device, model, criterion, optimizer)

        # Valid
        valid_loss_epoch, valid_accuracy_epoch = test(arr_valid_loader, device, model, criterion)
        valid_loss = valid_loss_epoch / len(valid_loader)

        nni.report_intermediate_result(valid_accuracy_epoch / len(valid_loader))
        print(
            f"Epoch [{epoch}/{n_epochs}]\t Train_Loss: {train_loss_epoch / len(train_loader)}\t Train_Accuracy: {train_accuracy_epoch / len(train_loader)}"
            f"\tValid_Loss: {valid_loss_epoch / len(valid_loader)}\t Valid_Accuracy: {valid_accuracy_epoch / len(valid_loader)}")

        if valid_loss < glob_loss:
            print('Valid loss decreased ({:.6f} --> {:.6f}).  Saving checkpoint ...'.format(glob_loss, valid_loss))
            glob_loss = valid_loss
            torch.save(model.state_dict(), 'classifier.pth')

    model.load_state_dict(torch.load('classifier.pth'))

    # final testing
    test_loss_epoch, test_accuracy_epoch = test(
        arr_test_loader, device, model, criterion
    )
    nni.report_final_result(test_accuracy_epoch / len(test_loader))
    print(
        f"[FINAL]\t Loss: {test_loss_epoch / len(test_loader)}\t Accuracy: {test_accuracy_epoch / len(test_loader)}"
    )


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        if(len(tuner_params) == 0):
            tuner_params = {"batch_size": 512, "lr": 0.001, "training_size": 0.8, "weight_decay": 1e-6}

        main(tuner_params)

    except Exception as exception:
        raise