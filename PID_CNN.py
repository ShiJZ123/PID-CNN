import torch
from torch import nn
from torchsummary import summary

class Concat(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.PRelu1 = nn.PReLU()
        self.PRelu2 = nn.PReLU()
        self.conv1 = nn.Conv3d(input_channels, input_channels, (1,3,3), 1, (0,1,1))
        self.conv2 = nn.Conv3d(input_channels, input_channels, (1,3,3), 1, (0,1,1))
        self.bn1 = nn.BatchNorm3d(input_channels)
        self.bn2 = nn.BatchNorm3d(input_channels)
        self.s = nn.AvgPool3d((1,2,2))

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.PRelu1(y)

        z = self.conv2(y)
        z = self.bn2(z)
        z = self.PRelu2(z)

        x = torch.concat([x,z], dim=1)
        x = self.s(x)
        return x

class PID_CNN(nn.Module):
    def __init__(self):
        super(PID_CNN,self).__init__()
        self.Con1 = Concat(2   )
        self.Con2 = Concat(4   )
        self.Con3 = Concat(8   )
        self.Con4 = Concat(16  )
        self.Con5 = Concat(32  )
        self.Con6 = Concat(64  )
        self.Con7 = Concat(128 )

        self.flatten = nn.Flatten()
        self.fp = nn.Linear(1024, 3)
        self.fv = nn.Linear(1024, 3)
        self.fa = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.Con1(x)
        x = self.Con2(x)
        x = self.Con3(x)
        x = self.Con4(x)
        x = self.Con5(x)
        x = self.Con6(x)
        x = self.Con7(x)

        p1 = x[:,:,0]
        p2 = x[:,:,1]
        p3 = x[:,:,2]

        p1 = self.flatten(p1)
        p2 = self.flatten(p2)
        p3 = self.flatten(p3)
        v_dif1 = p2 - p1
        v_dif2 = p3 - p2
        a_dif = v_dif2 -v_dif1

        p1 = self.fp(p1)  # input:[1536],output:[3]
        p2 = self.fp(p2)  # input:[1536],output:[3]
        p3 = self.fp(p3)  # input:[1536],output:[3]

        v1 = p2 -p1
        v2 = p3 -p2
        v_res1 = self.fv(v_dif1)
        v_res2 = self.fv(v_dif2)
        v1 = v1 + v_res1
        v2 = v2 + v_res2

        a = v2 - v1
        a_res = self.fa(a_dif)
        a = a + a_res

        result= torch.concat((p1,p2,p3,v1,v2,a),1)
        return result


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PID_CNN().to(device)

    x = torch.rand([4,2, 3, 256, 256]).to(device)
    y = model(x)
    # print(y)

    print(summary(model, (2, 3, 256, 256)))
