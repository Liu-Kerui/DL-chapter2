import math

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据预处理和加载
def load_data(batch_size=128):
    """
    加载FashionMNIST数据集并进行预处理
    """
    transform = transforms.Compose([
        transforms.Resize(128),  # 调整图像大小以适应VGG网络
        transforms.ToTensor(),   # 转换为PyTorch张量
    ])

    # 加载训练集和测试集
    train_dataset = datasets.FashionMNIST(
        root='../../data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(
        root='../../data', train=False, download=True, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# VGG网络块定义
def vgg_block(num_convs, in_channels, out_channels):
    """
    定义VGG网络的一个卷积块
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# VGG网络定义
class VGGNet(nn.Module):
    """
    定义VGG风格的卷积神经网络
    """
    def __init__(self, conv_arch):
        super(VGGNet, self).__init__()
        self.features = self._make_features(conv_arch)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_arch[-1][1] * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def _make_features(self, conv_arch):
        """
        根据架构定义构建特征提取部分
        """
        in_channels = 1  # FashionMNIST是单通道图像
        layers = []
        for (num_convs, out_channels) in conv_arch:
            layers.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 学习率调度器
class LearningRateScheduler:
    def __init__(self, optimizer, initial_lr, strategy, **kwargs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.strategy = strategy
        self.kwargs = kwargs
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.strategy == 'exponential_decay':
            decay_rate = self.kwargs.get('decay_rate')
            lr = self.initial_lr * (decay_rate ** self.epoch)
        elif self.strategy == 'warmup':
            warmup_epochs = self.kwargs.get('warmup_epochs')
            if self.epoch <= warmup_epochs:
                lr = self.initial_lr * (self.epoch / warmup_epochs)
            else:
                lr = self.initial_lr
        elif self.strategy == 'cyclic':
            period = self.kwargs.get('period')
            min_lr = self.kwargs.get('min_lr')
            max_lr = self.kwargs.get('max_lr')
            cycle = self.epoch % period
            lr = min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * cycle / period)) / 2
        else:
            raise ValueError("Unsupported strategy")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 训练过程
def train_model(net, train_loader, test_loader, num_epochs, lr, optimizer_type, **kwargs,): # train_model(net, train_loader, test_loader, num_epochs, lr, optimizer_type, strategy, **kwargs,)
    """
    训练模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    net.to(device)

    # 定义优化器 调度器和损失函数
    # if optimizer_type == 'SGD':
    #     optimizer = optim.SGD(net.parameters(), lr=lr)
    # elif optimizer_type == 'AdaGrad':
    #     optimizer = optim.Adagrad(net.parameters(), lr=lr)
    # elif optimizer_type == 'RMSprop':
    #     optimizer = optim.RMSprop(net.parameters(), lr=lr)
    # elif optimizer_type == 'AdaDelta':
    #     optimizer = optim.Adadelta(net.parameters(), lr=lr)
    # elif optimizer_type == 'Adam':
    #     optimizer = optim.Adam(net.parameters(), lr=lr)
    # else:
    #     raise ValueError("Unsupported optimizer type")

    # scheduler = LearningRateScheduler(optimizer, lr, strategy, **kwargs)

    # 定义优化器和损失函数
    if optimizer_type == 'SGD':
        momentum = kwargs.get('momentum')
        nesterov = kwargs.get('nesterov')
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer type")

    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        net.train()
        total_loss, total_correct = 0, 0

        # 训练循环
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (y_hat.argmax(1) == y).sum().item()

        # 更新学习率
        # scheduler.step()

        # 计算训练集准确率和损失
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / len(train_loader.dataset)

        # 评估测试集
        test_acc = evaluate_model(net, test_loader, device)

        # 记录结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}')

    # plot_results(train_losses, train_accs, test_accs, "VGG-8")
    return train_losses, train_accs, test_accs

# 评估模型
def evaluate_model(net, test_loader, device):
    """
    评估模型在测试集上的表现
    """
    net.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            correct += (y_hat.argmax(1) == y).sum().item()
    return correct / len(test_loader.dataset)

# 绘制训练曲线
def plot_results(results, title):  # train_losses, train_accs, test_accs, title
    """
    绘制训练损失和准确率曲线
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for method, (train_losses, train_accs, test_accs) in results.items():
        plt.plot(train_losses, label=method)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for method, (train_losses, train_accs, test_accs) in results.items():
        plt.plot(train_accs, label=f'{method} (Train)')
        plt.plot(test_accs, label=f'{method} (Test)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主函数
if __name__ == "__main__":
    # 超参数设置
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    conv_arch = [(1, 16), (1, 32), (2, 64), (2, 128)]  # VGG网络架构

    # 加载数据
    train_loader, test_loader = load_data(batch_size)

    # 创建网络
    net = VGGNet(conv_arch)
    print(net)

    # 创建一个随机输入张量
    X = torch.randn(size=(1, 1, 128, 128))  # (batch_size, channels, height, width)

    # 遍历特征提取部分（卷积部分）
    print("=== Feature Extraction Part ===")
    for blk in net.features:
        X = blk(X)
        print(f"{blk.__class__.__name__} output shape:\t{X.shape}")

    # 遍历分类器部分（全连接部分）
    print("\n=== Classifier Part ===")
    for blk in net.classifier:
        X = blk(X)
        print(f"{blk.__class__.__name__} output shape:\t{X.shape}")

    # # 实现不同的学习率调整策略
    # strategies = [
    #     ('exponential_decay', {'decay_rate': 0.9}),
    #     ('warmup', {'warmup_epochs': 5}),
    #     ('cyclic', {'period': 5, 'min_lr': 0.001, 'max_lr': 0.1})
    # ]
    #
    # for strategy, kwargs in strategies:
    #     print(f"\n=== Strategy: {strategy} ===")
    #     # 重新初始化网络
    #     net = VGGNet(conv_arch)
    #     # 训练模型
    # strategy, kwargs = ('cyclic', {'period': 5, 'min_lr': 0.001, 'max_lr': 0.1})
    # for optimizer_type in ("SGD","AdaGrad","RMSprop","AdaDelta","Adam"):
    #     train_model(net, train_loader, test_loader, num_epochs, learning_rate, optimizer_type, strategy, **kwargs)

    # 实验不同的梯度估计修正方法
    methods = [
        ('SGD', {'momentum': 0.9, 'nesterov': False}),
        ('Nesterov', {'momentum': 0.9, 'nesterov': True}),
        ('Adam', {'lr': 0.001}),
        ('Adam_low_lr', {'lr': 0.0001}),
        ('Adam_very_low_lr', {'lr': 0.00001}),
        ('SGD_with_grad_clip', {'momentum': 0.9, 'nesterov': False, 'grad_clip': 5.0})
    ]

    results = {}

    for method_name, params in methods:
        print(f"\n=== Method: {method_name} ===")
        # 创建网络
        net = VGGNet(conv_arch)
        # 训练模型并记录结果
        if method_name.startswith('Adam'):
            optimizer_type = 'Adam'
            lr = params['lr']
            train_losses, train_accs, test_accs = train_model(
                net, train_loader, test_loader, num_epochs, lr, optimizer_type)
        elif method_name == 'SGD_with_grad_clip':
            optimizer_type = 'SGD'
            train_losses, train_accs, test_accs = train_model(
                net, train_loader, test_loader, num_epochs, learning_rate, optimizer_type, **params)
        else:
            optimizer_type = 'SGD'
            train_losses, train_accs, test_accs = train_model(
                net, train_loader, test_loader, num_epochs, learning_rate, optimizer_type, **params)
        # 存储结果
        results[method_name] = (train_losses, train_accs, test_accs)

    # 绘制结果
    plot_results(results, 'VGG-8 with Different Gradient Estimation Methods')