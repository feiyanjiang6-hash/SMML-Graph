import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .gat_conv import GATConv

class SMMLGraph(torch.nn.Module):
    def __init__(self, in_dim1, in_dim2, hidden_dims):
        super(SMMLGraph, self).__init__()

        #self.alpha = alpha  # 模态间注意力权重
        # 模态1的编码器
        self.conv1_1 = GATConv(in_dim1, hidden_dims[0], heads=1, concat=False,
                               dropout=0, add_self_loops=False, bias=False)
        self.l2_1 = nn.Linear(hidden_dims[0],hidden_dims[1],bias=False)
        # self.conv1_2 = GATConv(hidden_dims[0], hidden_dims[1], heads=1, concat=False,
        #                        dropout=0, add_self_loops=False, bias=False)

        # 模态2的编码器
        self.conv1_2 = GATConv(in_dim2, hidden_dims[0], heads=1, concat=False,
                               dropout=0, add_self_loops=False, bias=False)
        self.l2_2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
        # self.conv2_2 = GATConv(hidden_dims[0], hidden_dims[1], heads=1, concat=False,
        #                        dropout=0, add_self_loops=False, bias=False)
        # 自适应融合权重
        # 使用线性层计算融合权重，确保权重在[0,1]范围内
        self.fusion_weight = nn.Sequential(
            nn.Linear(hidden_dims[1] * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # 共享解码器
        # self.decoder1 = GATConv(hidden_dims[1], hidden_dims[0], heads=1, concat=False,
        #                         dropout=0, add_self_loops=False, bias=False)
        self.decoder3_l = nn.Linear(hidden_dims[1], hidden_dims[0],bias=False)
        self.decoder4_1 = GATConv(hidden_dims[0], in_dim1, heads=1, concat=False,
                                  dropout=0, add_self_loops=False, bias=False)
        self.decoder4_2 = GATConv(hidden_dims[0], in_dim2, heads=1, concat=False,
                                  dropout=0, add_self_loops=False, bias=False)


    def forward(self, features1, features2, edge_index1,edge_index2):
        # edge_index1 表示空转构图， edge_index2 表示蛋白组构图
        h1_1 = F.elu(self.conv1_1(features1, edge_index1))
        h1_2 = F.elu(self.conv1_2(features2, edge_index2))
        z1 = self.l2_1(h1_1)
        z2 = self.l2_2(h1_2)

        # 自适应融合
        batch_size = z1.size(0)
        print(batch_size)
        # 计算每个样本的融合权重
        fusion_input = torch.cat([z1, z2], dim=1)
        alpha = self.fusion_weight(fusion_input)

        # 应用融合权重
        z = alpha * z1 + (1 - alpha) * z2
        #z = z1 * self.alpha + z2 * (1 - self.alpha)
        #解码
        h3 = F.elu(self.decoder3_l(z))
        # 分别解码到两个原始模态空间
        h4_1 = self.decoder4_1(h3,edge_index1)
        h4_2 = self.decoder4_2(h3,edge_index2)

        # 模态1编码
        # h1_1 = F.elu(self.conv1_1(features1, edge_index))
        # z1 = self.conv1_2(h1_1, edge_index, attention=False)
        #
        # # 模态2编码
        # h2_1 = F.elu(self.conv2_1(features2, edge_index))
        # z2 = self.conv2_2(h2_1, edge_index, attention=False)
        #
        # # 模态间注意力融合
        # alpha = torch.sigmoid(torch.tensor(self.alpha))  # 确保alpha在[0,1]范围内
        # z_fused = alpha * z1 + (1 - alpha) * z2
        #
        # # 共享解码器（使用融合后的表示）
        # h3 = F.elu(self.decoder1(z_fused, edge_index, attention=True,
        #                          tied_attention=self.conv1_1.attentions))
        #
        # # 分别解码到两个原始模态空间
        # out1 = self.decoder2_1(h3, edge_index, attention=False)
        # out2 = self.decoder2_2(h3, edge_index, attention=False)
        # z 融合嵌入，h4_1 空转重构, h4_2 蛋白组重构
        return z,h4_1,h4_2,alpha,z1,z2

if __name__ == '__main__':
    model = SMMLGraph(256,256,[128,64])
    x1 = torch.randn(*(100,256))
    x2 = torch.randn(*(100,256))
    e1 = torch.tensor([[1,2,3,4],[5,6,7,8]],dtype=torch.int64)
    e2 = torch.tensor([[1,2,3,4],[5,6,7,8]],dtype=torch.int64)
    with torch.no_grad():
        out = model.forward(x1,x2,e1,e2)
        print([o.shape for o in out])