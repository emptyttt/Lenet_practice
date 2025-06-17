from torch import nn

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