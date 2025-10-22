import torch
from torch import optim, nn

from inference.inference_method import InferenceAttentionMap
from utils.inference_utils import auc_metric


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(model, X_train, y_train, epochs=1000, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

def test_model(model, X_test, y_test):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        outputs=torch.softmax(outputs, dim=1)
        auc=auc_metric(y_test.cpu(), outputs.data.cpu())
        print(f'Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')
    return float(auc)

def train_and_test_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_auc = 0.0
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            outputs = torch.softmax(outputs[:,:len(torch.unique(y_test))], dim=1)
            auc = float(auc_metric(y_test.cpu(), outputs.data.cpu()))
            print(f'Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')
            if auc > best_auc:
                best_auc = auc
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Test AUC: {best_auc:.4f}')
        model.train()  # 设置模型为训练模式

    return best_auc

if __name__ == '__main__':
    pass