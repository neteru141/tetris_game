"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(32, 64), nn.ReLU(inplace=True))
        
        #in=(22,10,1) out=(24, 8, 8)
        self.conv2d_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channel=8, kernel_size=(1,5), stride=1, padding=1),
                                      nn.BatchNorm2d(8),
                                      nn.ReLU(inplace=True)
                                     )
        #in=out=(24,8,8)
        self.conv2d_2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channel=8, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(8),
                                      nn.ReLU(inplace=True)
                                     )
        
        #in=(24,8,8) out=(12,4,16)
        self.conv2d_3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channel=16, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2,2)
                                     )
        # in=out=(12,4,16)
        self.conv2d_4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channel=16, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True)
                                     )
        
        #in=(12,4,16) out=(6,2,32)
        self.conv2d_5 = nn.Sequential(nn.Conv2d(in_channels=16, out_channel=32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2,2)
                                     )
        
        # in=out=(6,2,32)
        self.conv2d_6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channel=32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True)
                                     )
        
        #in=(6,2,32) out=(3,1,64)
        self.conv2d_7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channel=64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2,2)
                                     )
        
        #in=(3,1,64) out=(1,1,64)
        self.conv2d_8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channel=64, kernel_size=(5,3), stride=1, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True)
                                     )
        
        self.outconv_1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.outconv_2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True))
        self.outconv_3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU(inplace=True))
        self.outconv_4 = nn.Sequential(nn.Linear(16, 1), nn.ReLU(inplace=True))

        #self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        y = self.conv2d_1(y)
        #y = self.conv2d_2(y)
        y = self.conv2d_3(y)
        #y = self.conv2d_4(y)
        y = self.conv2d_5(y)
        #y = self.conv2d_6(y)
        y = self.conv2d_7(y)
        y = self.conv2d_8(y)
        y = torch.flatten(y,1,2)
        
        z = torch.cat([x, y], dim=1)
        z = self.outconv_1(z)
        z = self.outconv_2(z)
        z = self.outconv_3(z)
        z = self.outconv_4(z)

        return z
