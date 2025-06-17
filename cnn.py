import torch
from torchvision import datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


#设置超参数（可调的参数）
learning_rate = 1e-3
batch_size = 64
epochs = 5

#下载训练和测试数据集并转化为张量形式
training_data = datasets.MNIST(
    root='data_mnist/train',
    train = True,
    transform=ToTensor(),
    download=True)

test_data = datasets.MNIST(
    root="data_mnist/test",
    train=False,
    download=True,
    transform=ToTensor()
)

#遍历 DataLoader
#由于我们指定了shuffle=True，因此在迭代所有批次后，数据会被打乱
'''
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
'''
#使用 DataLoaders 准备训练数据，这个就是切片分批次的数据
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
'''
#可视化数据
#print(len(training_data))#打印训练集中图像的数量。MNIST 一共有 60,000 张训练图片。
labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}
figure = plt.figure(figsize=(8, 8))#创建一个 8 英寸 × 8 英寸的画布，用于放多张图片。
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()#返回键值的元组
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''

'''
# 遍历Dataloader.
train_features, train_labels = next(iter(train_dataloader))#返回(X,y),X是一个批次上的形状大小[64,1,28,28]，y是标签大小[64]
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.title(label)
plt.show()
print(f"Label: {label}")
'''

#继承
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.LeNet = nn.Sequential(
            #第一层，卷积C1，ReLu
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.ReLU(),
            #第二层，池化S2
            nn.MaxPool2d(2,stride=2),
            #第三层，卷积层C2
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            #第四层，池化S2
            nn.MaxPool2d(2, stride=2),
            #第五层，卷积层C3
            nn.Conv2d(16, 120, kernel_size=5, padding=0),#这里还是4D
            nn.ReLU(),
        )
        self.fc=nn.Sequential(
            # 第六层，全连接层
            nn.Linear(120, 84),#只接受2D
            nn.ReLU(),
            # 第七层，全连接层
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x=self.LeNet(x)
        x = x.view(x.size(0), -1)  # 展平成一维向量
        x=self.fc(x)
        return x


train_losses = []
test_accuracies = []

#训练
def train_loop(dataloader, model, loss_fn, optimizer):
    #train_dataloader.dataset == training_data  # True
    #dataloader.dataset这个就是原始数据
    #img, label = train_dataloader.dataset[0]
    size = len(dataloader.dataset)
    total_loss = 0
    #将模型设置为训练模式——对于批量归一化和dropout层很重要
    #drop_out训练时随机丢一部分神经元（避免过拟合），测试时不丢。
    # Unnecessary in this situation but added for best practices
    #将模块设置为训练模式。这仅对某些模块有效。有关特定模块在训练 / 评估模式下的行为（即是否受影响），请参阅其文档，例如Dropout、BatchNorm等。
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #dataloader 是一个 可迭代对象，每次迭代都会返回一小批数据例如：第{0}批，[64,1,28,28]
        # Compute prediction and loss
        #forword
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward
        #通过调用反向传播预测损失loss.backward()。PyTorch 存储每个参数的损失梯度。
        loss.backward()
        #一旦我们有了梯度，我们就会调用optimizer.step()反向传播中收集的梯度来调整参数。
        optimizer.step()
        #重置模型参数的梯度。梯度默认累加；为了防止重复计算，我们在每次迭代时明确将其归零。
        optimizer.zero_grad()
        total_loss += loss.item()


        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)

#测试
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    #测试模式
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    #使用 torch.no_grad() 评估模型可确保在测试模式下不计算梯度 # 还可以减少不必要的梯度计算以及 require_grad=True 的张量的内存占用
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    test_accuracies.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

'''
#查看模型结构
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
'''
def plot_metrics():
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


'''
model = NeuralNetwork()  # 先新建一个同结构的模型
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()  # 设置为评估模式（关闭 dropout、batchnorm 的训练行为）
'''

if __name__ == "__main__":
    # 训练模型的代码放在这里
    model = NeuralNetwork()
    # Initialize the loss function(官方api)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 还有其他优化器如SGD
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    plot_metrics()
    print("Done!")
    # 保存权重
    torch.save(model.state_dict(), "model_weights.pth")
