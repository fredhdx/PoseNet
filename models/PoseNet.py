import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

from torchsummary import summary


def init(key, module, weights=None):
    if weights == None:
        return module
    # else:
    #     print(weights.keys())

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class MaxPoolReLU(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPoolReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()
        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # padding: (kernel - 1)/2
        # use ReLU: conv, maxpool, avgpool

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(f'inception_{key}/1x1',
                nn.Conv2d(in_channels, n1x1, kernel_size=1, stride=1, padding=0),
                weights
            ),
            nn.ReLU()
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init(f'inception_{key}/3x3_reduce',
                nn.Conv2d(in_channels, n3x3red, kernel_size=1, stride=1, padding=0),
                weights
            ),
            nn.ReLU(),
            init(f'inception_{key}/3x3',
                nn.Conv2d(n3x3red, n3x3, kernel_size=3, stride=1, padding=1),
                weights
            ),
            nn.ReLU()
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init(f'inception_{key}/5x5_reduce',
                nn.Conv2d(in_channels, n5x5red, kernel_size=1, stride=1, padding=0),
                weights
            ),
            nn.ReLU(),
            init(f'inception_{key}/5x5',
                nn.Conv2d(n5x5red, n5x5, kernel_size=5, stride=1, padding=2),
                weights
            ),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            init(f'inception_{key}/pool_proj',
                nn.Conv2d(in_channels, pool_planes, kernel_size=1, stride=1, padding=0),
                weights
            ),
            nn.ReLU()
        )

    def forward(self, x):

        # TODO: Feed data through branches and concatenate
        b1_out = self.b1(x)
        #print(f'inception box, b1_out: {b1_out.shape}')
        b2_out = self.b2(x)
        #print(f'inception box, b2_out: {b2_out.shape}')
        b3_out = self.b3(x)
        #print(f'inception box, b3_out: {b3_out.shape}')
        b4_out = self.b4(x)
        #print(f'inception box, b4_out: {b4_out.shape}')

        out = torch.cat([b1_out, b2_out, b3_out, b4_out], dim=1)

        return out


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()
        # TODO: Define loss headers
        
        if key == 1:
            self.loss_header = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
                nn.ReLU(),
                init('loss1/conv',
                     nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
                     weights),
                nn.ReLU(),
                nn.Flatten(),
                init('loss1/fc', nn.Linear(2048, 1024), weights),
                nn.Dropout(0.7)
            )
            self.fc_3 = nn.Linear(1024, 3)
            self.fc_4 = nn.Linear(1024, 4)
        elif key == 2:
            self.loss_header = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
                nn.ReLU(),
                init('loss2/conv',
                     nn.Conv2d(in_channels=528, out_channels=128, kernel_size=1, stride=1, padding=0),
                     weights),
                nn.ReLU(),
                nn.Flatten(),
                init('loss2/fc', nn.Linear(2048, 1024), weights),
                nn.Dropout(0.7)
            )
            self.fc_3 = nn.Linear(1024, 3)
            self.fc_4 = nn.Linear(1024, 4)
        else:
        # loss header 3
            self.loss_header = nn.Sequential(
                nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1024, 2048),
                nn.Dropout(p=0.4)
            )
            self.fc_3 = nn.Linear(2048, 3)
            self.fc_4 = nn.Linear(2048, 4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        x = self.loss_header(x)
        xyz = self.fc_3(x)
        wpqr = self.fc_4(x)

        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers
        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.001, beta=0.75, k=1),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            init('conv2/3x3',nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights)
        self._maxpoolrelu3 = MaxPoolReLU(3,2,1)

        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self._4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights)
        self._4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights)
        self._maxpoolrelu4 = MaxPoolReLU(3,2,1)

        self._5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights)
        self._5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights)


        self.loss1 = LossHeader(key=1, weights=weights)
        self.loss2 = LossHeader(key=2, weights=weights)
        self.loss3 = LossHeader(key=3)

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        #print(f'input size: {x.shape}')

        # sequence 1
        x = self.pre_layers(x)
        #print(f'pre layers: {x.shape}')
        x = self._3a(x)
        #print(f'_3a: {x.shape}')
        x = self._3b(x)
        #print(f'_3b: {x.shape}')
        x = self._maxpoolrelu3(x)
        x = self._4a(x)
        #print(f'_4a: {x.shape}')

        # loss 1
        loss1_xyz, loss1_wpqr = self.loss1(x)

        # sequence 2
        x = self._4b(x)
        #print(f'_4b: {x.shape}')
        x = self._4c(x)
        #print(f'_4c: {x.shape}')
        x = self._4d(x)
        #print(f'_4d: {x.shape}')

        # loss 2
        loss2_xyz, loss2_wpqr = self.loss2(x)

        # sequence 3
        x = self._4e(x)
        #print(f'_4e: {x.shape}')
        x = self._maxpoolrelu4(x)
        x = self._5a(x)
        #print(f'_5a: {x.shape}')
        x = self._5b(x)
        #print(f'_5b: {x.shape}')

        # loss 3
        loss3_xyz, loss3_wpqr = self.loss3(x)

        print(f'loss1_xyz', loss1_xyz.shape)
        print(f'loss1_wpqr', loss1_wpqr.shape)

        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):
    def __init__(self, w1_xyz, w2_xyz, w3_xyz, beta):
        super(PoseLoss, self).__init__()
        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.beta = beta


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss

        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        xyz_gt = poseGT[:, :3]
        wpqr_gt = poseGT[:, 3:]
        wpqr_gt_norm = F.normalize(wpqr_gt, p=2, dim=1)

        mse = torch.nn.MSELoss()

        # loss 1
        l1_xyz = mse(p1_xyz, xyz_gt)
        l1_wpqr = mse(p1_wpqr, wpqr_gt_norm) * self.beta

        # loss 2
        l2_xyz = mse(p2_xyz, xyz_gt)
        l2_wpqr = mse(p2_wpqr, wpqr_gt_norm) * self.beta

        # loss 3
        l3_xyz = mse(p3_xyz, xyz_gt)
        l3_wpqr = mse(p3_wpqr, wpqr_gt_norm) * self.beta

        loss = self.w1_xyz*(l1_xyz+l1_wpqr) + self.w2_xyz*(l2_xyz+l2_wpqr) + self.w3_xyz*(l3_xyz+l3_wpqr)
        return loss

        
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# posenet = PoseNet().to(device)
# summary(posenet, (3, 224, 224))