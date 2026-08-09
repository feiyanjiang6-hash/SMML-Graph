import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from .SMMLGraph import SMMLGraph
from .utils import Transfer_pytorch_Data
import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss


def train_SMMLGraph(adata1,adata2,hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='SMML-Graph',
                gradient_clipping=5,  weight_decay=0.0001, verbose=True,
                random_seed=0, save_loss=False, save_reconstruction=False,save_fusion_weights=True,
                contrastive_weight=0.1,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['SMML-Graph_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['SMML-Graph_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 确保输入数据是稀疏矩阵
    adata1.X = sp.csr_matrix(adata1.X)
    adata2.X = sp.csr_matrix(adata2.X)

    if 'highly_variable' in adata1.var.columns:
        adata1_Vars =  adata1[:, adata1.var['highly_variable']]
    else:
        adata1_Vars = adata1
    if 'highly_variable' in adata2.var.columns:
        adata2_Vars =  adata2[:, adata2.var['highly_variable']]
    else:
        adata2_Vars = adata2

    if verbose:
        print('Size of Input: ', adata1_Vars.shape)

    if 'Spatial_Net' not in adata1.uns.keys() or 'Spatial_Net' not in adata2.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    # 转换数据为PyTorch格式
    data1 = Transfer_pytorch_Data(adata1_Vars)
    data2 = Transfer_pytorch_Data(adata2_Vars)

    #model = SMMLGraph(hidden_dims = [data1.x.shape[1]] + hidden_dims).to(device)
    # 初始化模型 - 只需要一个模型实例
    model = SMMLGraph(
        in_dim1=data1.x.shape[1],  # 模态1的输入维度
        in_dim2=data2.x.shape[1],  # 模态2的输入维度
        hidden_dims=hidden_dims,  # 共享的隐藏层维度
    ).to(device)

    data1 = data1.to(device)
    data2 = data2.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 初始化融合权重记录列表
    fusion_weights = []  # 存储每个epoch的融合权重
    #loss_list = []
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        z, out1, out2,alpha,z1,z2 = model(data1.x, data2.x, data1.edge_index,data2.edge_index)
        loss1 = F.mse_loss(data1.x, out1)
        loss2 = F.mse_loss(data2.x, out2)
        reconstruction_loss = loss1 + loss2
        # 计算对比损失 - 使用余弦嵌入损失
        contrastive_loss = F.cosine_embedding_loss(
            z1,
            z2,
            # target=-torch.ones(z1.size(0)).to(device),
            # margin=0.5
            target=torch.ones(z1.size(0)).to(device),
            margin=0.0
        )

        # 总损失 = 重建损失 + 对比损失
        total_loss = reconstruction_loss + contrastive_weight * contrastive_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        # 记录融合权重
        if save_fusion_weights and epoch % 10 == 0:  # 每10个epoch记录一次
            fusion_weights.append(alpha.detach().cpu().numpy())

            # 打印训练进度
        if verbose and epoch % 100 == 0:
            print(f'Epoch {epoch}:')
            print(f'  Reconstruction Loss: {reconstruction_loss.item():.4f}')
            print(f'  Contrastive Loss: {contrastive_loss.item():.4f}')
            print(f'  Total Loss: {total_loss.item():.4f}')
            print(f'  Fusion Weight Mean: {alpha.mean().item():.4f}')
    
    model.eval()
    with torch.no_grad():
        z, _, _, alpha, _, _ = model(data1.x, data2.x, data1.edge_index,data2.edge_index)
#第一个返回值 z 是我们需要的嵌入表示第二个返回值 _ 表示我们不关心第一个模态的重建结果第三个返回值 _ 表示我们不关心第二个模态的重建结果
#第四个返回值 alpha 是我们需要的融合权重
    SMMLGraph_rep = z.to('cpu').detach().numpy()
    adata1.obsm[key_added] = SMMLGraph_rep
    adata2.obsm[key_added] = SMMLGraph_rep

    # 保存融合权重
    if save_fusion_weights:
        # 最后一个epoch的融合权重
        alpha_final = alpha.to('cpu').detach().numpy()
        adata1.obs['fusion_weight'] = alpha_final
        adata2.obs['fusion_weight'] = alpha_final

        # 所有记录的融合权重历史
        if fusion_weights:  # 只在列表不为空时保存
            adata1.uns['fusion_weight_history'] = np.array(fusion_weights)
            adata2.uns['fusion_weight_history'] = np.array(fusion_weights)

    if save_loss:
        adata1.uns['SMML-Graph_loss'] = loss.item()
        adata2.uns['SMML-Graph_loss'] = loss.item()

    if save_reconstruction:
        # 模态1的重建结果
        with torch.no_grad():
            _, out1, out2, _ = model(data1.x, data2.x, data1.edge_index, data2.edge_index)
        ReX1 = out1.to('cpu').detach().numpy()
        ReX1[ReX1 < 0] = 0
        adata1.layers['SMML-Graph_ReX'] = ReX1

        # 模态2的重建结果
        ReX2 = out2.to('cpu').detach().numpy()
        ReX2[ReX2 < 0] = 0
        adata2.layers['SMML-Graph_ReX'] = ReX2

    return adata1,adata2
