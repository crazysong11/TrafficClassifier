import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = torch.nn.Sequential(
            # 输入通道一定为1，输出通道为卷积核的个数，2为卷积核的大小（实际为一个[1,2]大小的卷积核）
            torch.nn.Conv1d(1, 8, 3),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(8, 16, 3),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool1d(4),
            torch.nn.Flatten(),
        )
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=48, out_features=1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.model1(input)
        x = self.model2(x)
        return x