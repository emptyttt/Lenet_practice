from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

#下载训练和测试数据集并转化为张量形式
def get_dataloaders(batch_size):
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
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader,test_dataloader