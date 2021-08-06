# masterzero-hw-ssl-simclr
- Dataset: cifar10 

## Requirement
`pip install simclr`

## SimCLR Model
- Backbone: modified resnet18
- Projector: hidden_dim=512, projection_dim=128
    ```python=
    self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.hidden_dim, bias=False),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, projection_dim, bias=True),
            )
    ```
### Transform
- Modified from simclr.modules.transformations.TransformsSimCLR
    ```python=
        s = 0.5
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                )
            ]
        )
    ```
### SimCLR Training Setting
- Without label
- batch_size = 512
- epochs = 500
- optimizer
  - `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)`
- loss
  - `NT_Xent(batch_size, temperature=0.5, world_size=1)`

## Classifier Model
- hidden = 20480
- n_classes = 10
```python
class LogisticRegression(nn.Module):
    def __init__(self, n_features, hidden, n_classes):
        super(LogisticRegression, self).__init__()

        self.linear1 = nn.Linear(n_features, hidden)
        self.drop = nn.Dropout(0.8)
        self.linear2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.drop(x)
        return self.linear2(x)
```
### Classifier Training Setting
- With full data label
- batch_size = 512
- epochs = 150
- optimizer
  - SGD
  - lr = 0.2
  - momentum = 0.9
- scheduler
  - LambdaLR

## Experiment Result
### TOP1
```bash
Top 1 Accuracy of class  0 is 925/1000  92.50%
Top 1 Accuracy of class  1 is 958/1000  95.80%
Top 1 Accuracy of class  2 is 852/1000  85.20%
Top 1 Accuracy of class  3 is 768/1000  76.80%
Top 1 Accuracy of class  4 is 886/1000  88.60%
Top 1 Accuracy of class  5 is 788/1000  78.80%
Top 1 Accuracy of class  6 is 916/1000  91.60%
Top 1 Accuracy of class  7 is 909/1000  90.90%
Top 1 Accuracy of class  8 is 954/1000  95.40%
Top 1 Accuracy of class  9 is 934/1000  93.40%
Top 1 accuracy of the network on the 10000 test images: 8890/10000  88.90 %
```
### TOP3
```
Top 3 Accuracy of class  0 is 991/1000  99.10%
Top 3 Accuracy of class  1 is 996/1000  99.60%
Top 3 Accuracy of class  2 is 977/1000  97.70%
Top 3 Accuracy of class  3 is 975/1000  97.50%
Top 3 Accuracy of class  4 is 977/1000  97.70%
Top 3 Accuracy of class  5 is 972/1000  97.20%
Top 3 Accuracy of class  6 is 974/1000  97.40%
Top 3 Accuracy of class  7 is 982/1000  98.20%
Top 3 Accuracy of class  8 is 991/1000  99.10%
Top 3 Accuracy of class  9 is 991/1000  99.10%
Top 3 accuracy of the network on the 10000 test images: 9826/10000  98.26 %
```
