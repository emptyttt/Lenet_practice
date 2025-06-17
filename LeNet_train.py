from LeNet_model import NeuralNetwork
import torch
from torch import nn, optim
from data_loader import get_dataloaders
from LeNet_utils import train_loop,test_loop,plot_metrics

#设置超参数（可调的参数）
learning_rate = 1e-3
batch_size = 64
epochs = 5
#设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

train_dataloader,test_dataloader=get_dataloaders(batch_size=batch_size)

if __name__ == "__main__":
    # 训练模型的代码放在这里
    model = NeuralNetwork().to(device)
    # Initialize the loss function(官方api)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 还有其他优化器如SGD
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer,batch_size=batch_size)
        test_loop(test_dataloader, model, loss_fn)
    plot_metrics()
    print("Done!")
    # 保存权重
    torch.save(model.state_dict(), "model_weights.pth")