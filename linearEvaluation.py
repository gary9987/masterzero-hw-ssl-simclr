from simclrModel import SimCLR
from simclr.modules import LogisticRegression
from torch.utils.data import DataLoader
import torchvision
from simclr.modules.transformations import TransformsSimCLR
import torch
from tqdm import tqdm
import numpy as np
from simclr.modules.resnet_hacks import modify_resnet_model

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


def get_features(context_model, train_loader, test_loader, device):

    train_X, train_y = inference(train_loader, context_model, device)
    test_X, test_y = inference(test_loader, context_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):

    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = torchvision.datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=TransformsSimCLR(size=32).test_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        './data',
        train=False,
        download=True,
        transform=TransformsSimCLR(size=32).test_transform
    )

    batch_size = 512

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    model_fp = 't0.5checkpoint1.pth'
    simclr_model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
    simclr_model = simclr_model.to(device)
    simclr_model.eval()

    ## Logistic Regression (classifier)
    n_classes = 10  # stl-10 / cifar-10
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, train_loader, test_loader, device
    )


    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, batch_size
    )

    n_epochs = 500
    glob_loss = np.Inf
    for epoch in range(1, n_epochs+1):
        loss_epoch, accuracy_epoch = train(arr_train_loader, device, model, criterion, optimizer)
        loss = loss_epoch / len(train_loader)
        print(f"Epoch [{epoch}/{n_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}")

        if loss < glob_loss:
            print('Training loss decreased ({:.6f} --> {:.6f}).  Saving checkpoint ...'.format(glob_loss, loss))
            glob_loss = loss
            torch.save(model.state_dict(), 'classifier.pth')

    model.load_state_dict(torch.load('classifier.pth'))

    # final testing
    loss_epoch, accuracy_epoch = test(
        arr_test_loader, device, model, criterion
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
    )