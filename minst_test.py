import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn import NeuralNetwork

# 加载训练好的模型
model = NeuralNetwork()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()  #设置为评估模式

test_dataset = datasets.MNIST(root="data_mnist/test", train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#进行一次推理
with torch.no_grad():  # 不计算梯度
    for X, y in test_loader:
        output = model(X)  # 前向传播
        predicted = torch.argmax(output, dim=1)  # 取出概率最大的类别
        print(f"预测结果：{predicted.item()}，真实标签：{y.item()}")
        break  # 这里只推理一张
