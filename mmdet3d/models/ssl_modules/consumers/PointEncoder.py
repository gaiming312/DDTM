from ...builder import SSL_MODULES
import torch
import torch.nn as nn
from ..utils import mlvl_get,mlvl_set,mlvl_getattr
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class PointNetEncoder(nn.Module):
    def __init__(self, reweight_matrix=True,channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc = nn.Linear(1024,128)
        # self.global_feat = global_feat
        # self.feature_transform = feature_transform
        self.reweight_matrix = reweight_matrix

    def forward(self, x):

        B, D, N = x.size() # batch_size (x,y,z) num_points
        x = x.permute(0,2,1)
        if D > 3: # 当不止3维 ， 有其他信息加入时
            feature = x[:, :, 3:]  # 其他特征
            x = x[:, :, :3] # x
        x = x.permute(0,2,1)
        trans = self.stn(x)    # 得到变换矩阵 24，3，3
        x = x.transpose(2, 1) # 24 ， 1024， 3
        x = torch.bmm(x, trans) # perform 矩阵乘法
        # if D > 3:
        #     x = torch.cat([x, feature], dim=2) # 因为转置了所以在最后一维度
        x = x.transpose(2, 1) # 转回来
        x = F.relu(self.bn1(self.conv1(x))) # 24，3，1024


        pointfeat = x # 现在点云特征就提取完毕
        x = F.relu(self.bn2(self.conv2(x))) # 24 ，128，1024
        x = self.bn3(self.conv3(x)) # 24，1024，1024
        x = torch.max(x, 2, keepdim=True)[0] # 24，1024，1 max pooling
        x = x.view(-1, 1024) # 24,1024
        # if self.global_feat:
        #     return x, trans  # （24,1024） （24,3,3）
        if self.reweight_matrix:
            return self.fc(x) # (N,128)
        else:
            return x



@SSL_MODULES.register_module()
class ReweightModule():
    def __init__(self,pos_point_key,out_key,checkpoints=None):
        self.pos_point_key = pos_point_key
        self.out_key = out_key
        self.checkpoints = checkpoints

        self.Encoder = PointNetEncoder()
        if self.checkpoints is not None:
            self.Encoder.load_state_dict(torch.load(self.checkpoints,map_location=
                                                    torch.device('cpu')),strict=False,)

    def forward(self,ssl_object,batch_dict):

        x = mlvl_get(batch_dict,self.pos_point_key)
        x = [point[:,:3] for point in x]
        x = pad_sequence(x,batch_first=True).permute(0,2,1)
        batch, _, __ = x.size()
        if batch == 1 :
            self.Encoder.apply(set_bn_eval)
        self.Encoder.to(x.device)
        out = self.Encoder(x)

        mlvl_set(batch_dict,self.out_key,out.unsqueeze(dim=1))

        """
        try to save checkpoint by using ssl_object.epoch
        
        add save check point to Point Encoder
        """
        # if ssl_object.epoch ==40:
        #     torch.save(self.Encoder.state_dict(),\
        #     "/home/lab/YYF/PointCloud/DetMatch/outputs/encoder/encoder.pth")

        return batch_dict



class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0] # 读取batch size
        x = F.relu(self.bn1(self.conv1(x))) # 24，64，1024
        x = F.relu(self.bn2(self.conv2(x))) # 24，128，1024
        x = F.relu(self.bn3(self.conv3(x))) # 24，1024，1024
        x = torch.max(x, 2, keepdim=True)[0] # 24，1024，1 提取最大的点？
        x = x.view(-1, 1024) # 24，1024

        x = F.relu(self.bn4(self.fc1(x))) # 24，512
        x = F.relu(self.bn5(self.fc2(x))) # 24 ，256
        x = self.fc3(x) # 24，9

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1) # 24，9
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3) # 24，3，3
        return x # 这里是预测一个变换矩阵

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

if __name__ == "__main__":
    from torch.nn.utils.rnn import pad_sequence
    net = PointNetEncoder()
    cp_path = "/home/lab/YYF/PointCloud/DetMatch/tools/pretrain_encoder/checkpoints/FPS_sampled/encoder_0.01_0.pth"
    net.load_state_dict(torch.load(cp_path),strict=False)
    print("successful load")


    # out = net(test)
    # print(out.shape,test.shape)