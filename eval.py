import linearEvaluation
from torch.optim.lr_scheduler import LambdaLR
from simclrModel import SimCLR
from linear import LogisticRegression
from torch.utils.data import DataLoader
import torchvision
from simclr.modules.transformations import TransformsSimCLR
import torch
from tqdm import tqdm
import numpy as np
from simclr.modules.resnet_hacks import modify_resnet_model
import torch.utils.data as data

def evaluteTopK(k, model, loader):
    model.eval()

    class_correct = [0. for i in range(10)]
    class_total = [0. for i in range(10)]

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            y_resize = labels.view(-1, 1)
            _, predicted = outputs.topk(k, 1, True, True)

            for i in range(len(predicted)):
                class_total[labels[i]] += 1
                #print(torch.eq(predicted[i], y_resize[i]).sum().float().item())
                class_correct[labels[i]] += torch.eq(predicted[i], y_resize[i]).sum().float().item()

    for i in range(10):
        print('Top %d Accuracy of class %2d is %3d/%3d  %.2f%%' % (
            k, i, class_correct[i], class_total[i], (100 * class_correct[i] / class_total[i])))

    print('Top %d accuracy of the network on the %d test images: %d/%d  %.2f %%'
          % (k, sum(class_total), sum(class_correct), sum(class_total), (100 * sum(class_correct) / sum(class_total))))

    return 100 * sum(class_correct) / sum(class_total)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_dataset = torchvision.datasets.CIFAR10(
        './data',
        train=False,
        download=True,
        transform=TransformsSimCLR(size=32).test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )

    encoder = modify_resnet_model(torchvision.models.resnet18())
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, hidden_dim=512, projection_dim=128, n_features=n_features)
    simclr_model.load_state_dict(torch.load('checkpoint1.pth', map_location='cuda:0'))
    simclr_model.to(device)
    simclr_model.eval()

    ## Logistic Regression (classifier)
    n_classes = 10  # stl-10 / cifar-10
    model = LogisticRegression(simclr_model.n_features, 20480, n_classes)
    model.load_state_dict(torch.load('classifier.pth', map_location='cuda:0'))
    model = model.to(device)
    model.eval()

    print("### Creating features from pre-trained context model ###")
    (_, _, _, _, test_X, test_y) = linearEvaluation.get_features(
        simclr_model, test_loader, test_loader, test_loader, device
    )

    _, _, arr_test_loader = linearEvaluation.create_data_loaders_from_arrays(
        test_X, test_y, test_X, test_y, test_X, test_y, 512
    )

    criterion = torch.nn.CrossEntropyLoss()
    # final testing
    test_loss_epoch, test_accuracy_epoch = linearEvaluation.test(
        arr_test_loader, device, model, criterion
    )
    print(
        f"[FINAL]\t Loss: {test_loss_epoch / len(test_loader)}\t Accuracy: {test_accuracy_epoch / len(test_loader)}"
    )

    evaluteTopK(1, model, arr_test_loader)
    evaluteTopK(3, model, arr_test_loader)