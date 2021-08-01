import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.linear1 = nn.Linear(n_features, 2048)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.drop(x)
        return self.linear2(x)
