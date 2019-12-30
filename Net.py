import torch
import torch.nn as nn

#正视图  输入1*576*720，输出1*36   
class MyNet_Front(nn.Module):
    def __init__(self):
        super(MyNet_Front, self).__init__()
        self.small = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2))
            )

        self.min = nn.MaxPool2d((1,45))
         
    def forward(self, x):
        x = self.small(x)
        x = -x
        x = -self.min(x)            #x = - max_pool(-x)   最小池化
        x = x.view(x.size(0), -1)
        return x


#侧视图  输入1*576*720，输出1*36
class MyNet_Lateral(nn.Module):
    def __init__(self):
        super(MyNet_Lateral, self).__init__()
        self.small = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2))
            )

        self.fc = nn.Sequential(
            nn.Linear(1*36*45, 1*36),
            nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.small(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#主视侧视图参数关联 输入主视图1*36 侧视图1*36 输出关联系数1*36
class MyNet_Correlation(nn.Module):
    def __init__(self):
        super(MyNet_Correlation, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1*72, 1*36),
            nn.LeakyReLU(0.2, True)
            )

    def forward(self, img_F, img_L):
        data_correlation = torch.cat((img_F, img_L), 1) 
        data_correlation = self.fc(data_correlation)
        return data_correlation

#三个神经网络整合
class MyNet_All(nn.Module):
    def __init__(self):
        super(MyNet_All, self).__init__()
        self.small = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,2))
            )

        self.min = nn.MaxPool2d((1,45))

        self.fc_L = nn.Sequential(
            nn.Linear(1*36*45, 1*36),
            nn.LeakyReLU(0.2, True)
            )

        self.fc_C = nn.Sequential(
            nn.Linear(1*72, 1*36),
            nn.LeakyReLU(0.2, True)
            )

    def forward(self, img_F, img_L):
        img_F = self.small(img_F)
        img_F = -img_F
        img_F = -self.min(img_F)            #x = - max_pool(-x)   最小池化
        img_F = img_F.view(img_F.size(0), -1)

        img_L = self.small(img_L)
        img_L = img_L.view(img_L.size(0), -1)
        img_L = self.fc_L(img_L)

        data_correlation = torch.cat((img_F, img_L), 1) 
        data_correlation = self.fc_C(data_correlation)
        return data_correlation



#神经网络
class MyNet_New(nn.Module):
    def __init__(self):
        super(MyNet_New, self).__init__()
        self.small = nn.Sequential(
            nn.Conv2d(1, 3, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
       #    nn.AvgPool2d((2,2)),

            nn.Conv2d(3, 2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(2, 1, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            )

        #self.FC = nn.Sequential(
        #    nn.Linear(1*36*45, 1*36),
        #    nn.LeakyReLU(0.2, True)
        #    )

        self.fc_C = nn.Sequential(
            nn.Linear(2*15*25, 25*25*15),
            nn.LeakyReLU(0.2, True),
            nn.Linear(25*25*15, 2),
            nn.LeakyReLU(0.2, True),
            )

    def forward(self, img_F, img_L):
        img_F = self.small(img_F)
        img_F = img_F.view(img_F.size(0), -1)
    #    img_F = self.FC(img_F)

        img_L = self.small(img_L)
        img_L = img_L.view(img_L.size(0), -1)
    #    img_L = self.FC(img_L)

        data_correlation = torch.cat((img_F, img_L), 1) 
        data_correlation = self.fc_C(data_correlation)
        return data_correlation

