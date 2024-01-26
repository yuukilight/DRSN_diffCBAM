# yuukilight
from module_YJC import *

class DNN(torch.nn.Module):
    def __init__(self, input_size = 8192):
        super().__init__()
        self.input_size = input_size
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(512, 64)
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=4)
        )
    def forward(self, input):  #
        # x = self.bn(input)  #
        x = self.fc1(input)
        x = self.relu(x)
        x = self.bn(x)
        out = self.output(x)
        return out

class CNN(torch.nn.Module):
    def __init__(self, input_size = 8192):
        super().__init__()
        self.input_size = input_size
        self.bn1 = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=4, padding=calculate_padding(input_size, 5, 4)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=calculate_padding(input_size // 4, 5, 2)),
            nn.ReLU()
        )
        self.curl = input_size // 8
        self.conv2 = nn.Sequential (
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=4,
                      padding=calculate_padding(self.curl, 5, 4)),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=4,
                      padding=calculate_padding(self.curl // 4, 5, 4)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1,
                      padding=calculate_padding(self.curl // 32, 3, 2)),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 16, 3, 2)),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                      padding=calculate_padding(self.curl // 32, 3, 2)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                  padding=calculate_padding(self.curl // 32, 3, 2)),
            nn.ReLU()
        )
        self.curl = self.curl // 64
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.curl = self.curl * 16
        self.output = nn.Sequential(
            nn.Linear(self.curl, self.curl * 2),
            nn.Linear(self.curl * 2, out_features=4)
        )
    def forward(self, input):  #
        x = self.bn1(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.flatten(x)  # 256
        out = self.output(x)  # 1*4
        return out


class ResNet(torch.nn.Module):
    def __init__(self, input_size = 8192):
        super().__init__()
        self.input_size = input_size
        self.bn1 = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=4, padding=calculate_padding(input_size, 5, 4)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=calculate_padding(input_size // 2, 5, 2))
        )
        self.curl = input_size // 8
        self.res1 = nn.Sequential(
            RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSU(in_channels=16, out_channels=32, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2,
                      padding=calculate_padding(input_size // 2, 5, 2)),
            RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSU(in_channels=16, out_channels=32, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2,
                      padding=calculate_padding(input_size // 2, 5, 2)),
            RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True)
        )
        self.bn = nn.BatchNorm1d(16)
        self.curl = self.curl // 4
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(self.curl, self.curl * 2),
            nn.Linear(self.curl * 2, out_features=4)
        )

    def forward(self, input):  #
        x = self.bn1(input)
        x = self.conv1(x)  # /8 1024
        x = self.res1(x)
        x = self.bn(x)  # 16*16
        x = self.relu(x)
        x = self.flatten(x)  # 256
        out = self.output(x)  # 64
        return out



class DRSN(torch.nn.Module):
    def __init__(self, input_size=8192):
        super().__init__()
        self.input_size = input_size
        self.bn1 = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=4,
                      padding=calculate_padding(input_size, 5, 4)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2,
                      padding=calculate_padding(input_size // 2, 5, 2))
        )
        self.curl = input_size // 8
        self.ress1 = nn.Sequential(
            RSSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSSU(in_channels=16, out_channels=32, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2,
                      padding=calculate_padding(input_size // 2, 5, 2)),
            RSSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSSU(in_channels=16, out_channels=32, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2,
                      padding=calculate_padding(input_size // 2, 5, 2)),
            RSSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True)
        )
        self.bn = nn.BatchNorm1d(16)
        self.curl = self.curl // 4
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(self.curl, self.curl * 2),
            nn.Linear(self.curl * 2, out_features=4)
        )

    def forward(self, input):  #
        x = self.bn1(input)
        x = self.conv1(x)  # /8 1024
        x = self.ress1(x)
        x = self.bn(x)  # 16*16
        x = self.relu(x)
        x = self.flatten(x)  # 256
        out = self.output(x)  # 64
        return out


class DRSN_1D_CBAM_v2(torch.nn.Module):
    def __init__(self, input_szie = 16384):
        super().__init__()
        self.input_size = input_szie // 2
        self.bn1 = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=4, padding=calculate_padding(input_szie, 5, 4)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=calculate_padding(input_szie // 4, 5, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=4, padding=calculate_padding(input_szie, 5, 4)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=calculate_padding(input_szie // 4, 5, 2))
        )
        self.curl = self.input_size // 8
        self.ress1 = nn.Sequential(
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True),
            RSSU(in_channels=32, out_channels=64, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 2, 3, 2)),
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True),
            RSSU(in_channels=32, out_channels=64, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 2, 3, 2)),
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=False),
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True)
        )
        # self.ress2 = nn.Sequential(
        #     RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
        #     RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
        #     RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
        #     RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
        #     RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
        #     RSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True)
        # )
        self.curl = self.curl // 32
        self.cbam = Cbam(32)
        self.curl = self.curl * 32
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(self.curl, self.curl // 2),
            nn.Linear(self.curl // 2, self.curl * 2),
            nn.Linear(self.curl * 2, out_features=4)
        )

    def forward(self, input):  #
        b,c,h = input.size()
        x = input.view([b, 2, h//2])
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        x1 = self.bn1(x1.view([b, 1, h // 2]))
        x2 = self.bn1(x2.view([b, 1, h // 2]))
        x1 = self.conv1(x1)  # /8 1024
        x2 = self.conv2(x2)  # /8 1024
        x = torch.cat((x1, x2), dim=1)
        x = self.ress1(x)
        x = self.cbam(x)
        x = self.bn(x)  # 16*16
        x = self.relu(x)
        x = self.flatten(x)  # 256
        out = self.output(x)  # 64
        return out


class DRSN_diffCBAM(torch.nn.Module):
    def __init__(self, input_szie = 16384):
        super().__init__()
        self.input_size = input_szie // 2
        self.bn1 = nn.BatchNorm1d(2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, stride=4, padding=calculate_padding(input_szie, 5, 4)),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=calculate_padding(input_szie // 4, 5, 2))
        )
        self.curl = self.input_size // 8
        self.ress1 = nn.Sequential(
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True),
            RSSU(in_channels=32, out_channels=64, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 2, 3, 2)),
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True),
            RSSU(in_channels=32, out_channels=64, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 2, 3, 2)),
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True),
            RSSU(in_channels=32, out_channels=64, kernel_size=3, down_sample=True),
            RSSU(in_channels=64, out_channels=128, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 2, 3, 2)),
            RSSU(in_channels=64, out_channels=128, kernel_size=3, down_sample=True)
        )
        self.curl = self.curl // 256
        self.cbam = Cbam(128)
        self.curl = self.curl * 128
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(self.curl, self.curl // 2),
            nn.Linear(self.curl // 2, self.curl * 2),
            nn.Linear(self.curl * 2, out_features=4)
        )

    def forward(self, input):  #
        b,c,h = input.size()
        x = self.bn1(input.view([b, 2, h//2]))
        x = self.conv1(x)  # /8 1024
        x = self.ress1(x)
        x = self.cbam(x)
        x = self.bn(x)  # 16*16
        x = self.relu(x)
        x = self.flatten(x)  # 256
        out = self.output(x)  # 64
        return out

class DRSN_diff(torch.nn.Module):
    def __init__(self, input_szie = 16384):
        super().__init__()
        self.input_size = input_szie // 2
        self.bn1 = nn.BatchNorm1d(2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, stride=4, padding=calculate_padding(input_szie, 5, 4)),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=calculate_padding(input_szie // 4, 5, 2))
        )
        self.curl = self.input_size // 8
        self.ress1 = nn.Sequential(
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True),
            RSSU(in_channels=32, out_channels=64, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 2, 3, 2)),
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True),
            RSSU(in_channels=32, out_channels=64, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 2, 3, 2)),
            RSSU(in_channels=32, out_channels=32, kernel_size=3, down_sample=True),
            RSSU(in_channels=32, out_channels=64, kernel_size=3, down_sample=True),
            RSSU(in_channels=64, out_channels=128, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2,
                      padding=calculate_padding(self.curl // 2, 3, 2)),
            RSSU(in_channels=64, out_channels=128, kernel_size=3, down_sample=True)
        )
        self.curl = self.curl // 256
        self.cbam = Cbam(128)
        self.curl = self.curl * 128
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(self.curl, self.curl // 2),
            nn.Linear(self.curl // 2, self.curl * 2),
            nn.Linear(self.curl * 2, out_features=4)
        )

    def forward(self, input):  #
        b,c,h = input.size()
        x = self.bn1(input.view([b, 2, h//2]))
        x = self.conv1(x)  # /8 1024
        x = self.ress1(x)
        # x = self.cbam(x)
        x = self.bn(x)  # 16*16
        x = self.relu(x)
        x = self.flatten(x)  # 256
        out = self.output(x)  # 64
        return out



class DRSN_CBAM(torch.nn.Module):
    def __init__(self, input_size=8192):
        super().__init__()
        self.input_size = input_size
        self.bn1 = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=4,
                      padding=calculate_padding(input_size, 5, 4)),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2,
                      padding=calculate_padding(input_size // 2, 5, 2))
        )
        self.curl = input_size // 8
        self.ress1 = nn.Sequential(
            RSSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSSU(in_channels=16, out_channels=32, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2,
                      padding=calculate_padding(input_size // 2, 5, 2)),
            RSSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSSU(in_channels=16, out_channels=32, kernel_size=3, down_sample=False),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2,
                      padding=calculate_padding(input_size // 2, 5, 2)),
            RSSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True),
            RSSU(in_channels=16, out_channels=16, kernel_size=3, down_sample=True)
        )
        self.cbam = Cbam(16)
        self.bn = nn.BatchNorm1d(16)
        self.curl = self.curl // 4
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(self.curl, self.curl * 2),
            nn.Linear(self.curl * 2, out_features=4)
        )

    def forward(self, input):  #
        x = self.bn1(input)
        x = self.conv1(x)  # /8 1024
        x = self.ress1(x)
        x = self.cbam(x)
        x = self.bn(x)  # 16*16
        x = self.relu(x)
        x = self.flatten(x)  # 256
        out = self.output(x)  # 64
        return out


if __name__ == '__main__':
    model = DRSN_1D_CBAM_v2()
    model.cuda()
    model.double()
    x = torch.rand(6, 1, 8192*2)
    x = x.cuda()
    x = x.double()
    out = model(x)
    print(out.size())