import torch.nn as nn
import torch.nn.functional as F

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
