import torch
import matplotlib.pyplot as plt

train_losses = []
test_accuracies = []


#训练
def train_loop(dataloader, model, loss_fn, optimizer,batch_size):
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