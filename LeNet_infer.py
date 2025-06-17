import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from LeNet_model import NeuralNetwork
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for inference")

# 加载模型结构并加载权重
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
model.eval()  # 设置为推理模式

# 加载测试数据
test_dataset = datasets.MNIST(
    root="data_mnist/test",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 推理一张图片
with torch.no_grad():  # 不计算梯度
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        predicted = torch.argmax(output, dim=1)
        print(f"预测结果：{predicted.item()}，真实标签：{y.item()}")
        break  # 只看一张图


plt.imshow(X.cpu().squeeze(), cmap="gray")
plt.title(f"预测：{predicted.item()}，真实：{y.item()}")
plt.axis("off")
plt.show()
