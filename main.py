from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn
import torch
import os
from Captcha_Data_Set import one_hot_encode as encode
from Captcha_Data_Set import String, CHAR_NUMBER

BATCH_SIZE = 60
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class ImageDataSet(Dataset):
    def __init__(self, dir_path):
        super(ImageDataSet, self).__init__()
        self.img_path_list = [f"{dir_path}/{filename}" for filename in os.listdir(dir_path)]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale()
        ])

    def __getitem__(self, idx):
        image = self.trans(Image.open(self.img_path_list[idx]))
        label = self.img_path_list[idx].split("/")[-1].replace(".png", "")
        # print(self.img_path_list[idx])
        # print(label)
        label = encode(label)
        return image, label

    def __len__(self):
        return len(self.img_path_list)


# 加载数据
def get_loader(path):
    dataset = ImageDataSet(path)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    return dataloader


# if __name__ == "__main__":
#     train_dataloader = get_loader("./data/train")
#     test_dataloader = get_loader("./data/test")
#     for X, y in train_dataloader:
#         print(X.shape)
#         print(y.shape)
#         break


# 网络搭建
class NeuralNetWork(nn.Module):
    def __init__(self):
        super(NeuralNetWork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
            nn.AvgPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
            nn.AvgPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
            nn.AvgPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
            nn.AvgPool2d(2)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=15360, out_features=4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=CHAR_NUMBER * len(String))
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


# 训练，没啥好说的
def train(dataloader, model, loss_fn, optimizer):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"损失值: {loss:>7f}")


# 损失函数使用的是MultiLabelSoftMarginLoss，优化器用的是Adam
def main():
    model = NeuralNetWork().to(device)
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = get_loader("./data/clean_data/")
    # 先设了个25，看看效果啥样的，看看什么时候无变化
    # 偶尔震荡，在大一点点看看
    epoch = 25
    for t in range(epoch):
        print(f"训练周期 {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print("\n")
    # 保存模型下来
    torch.save(model.state_dict(), "model6.pth")
    print("训练完成，模型已保存")


if __name__ == "__main__":
    main()
