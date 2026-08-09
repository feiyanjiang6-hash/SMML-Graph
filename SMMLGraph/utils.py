import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial import distance_matrix
from collections import Counter

import torch
from torch_geometric.data import Data

def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data

def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y'], plot_Stats=False):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1/num_batch_x)*x*100) for x in range(num_batch_x+1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1/num_batch_y)*x*100) for x in range(num_batch_y+1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x+1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y+1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list

# def lsi(
#         adata: anndata.AnnData, n_components: int = 20,
#         use_highly_variable: Optional[bool] = None, **kwargs
#        ) -> None:
#     r"""
#     LSI analysis (following the Seurat v3 approach)
#     """
#     if use_highly_variable is None:
#         use_highly_variable = "highly_variable" in adata.var
#     adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
#     X = tfidf(adata_use.X)
#     #X = adata_use.X
#     X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
#     X_norm = np.log1p(X_norm * 1e4)
#     X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
#     X_lsi -= X_lsi.mean(axis=1, keepdims=True)
#     X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
#     #adata.obsm["X_lsi"] = X_lsi
#     adata.obsm["X_lsi"] = X_lsi[:,1:]

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net_local'] = Spatial_Net


def Cal_Gene_Similarity_Net_KNN(adata, k_neighbors=6, metric='cosine', verbose=True):
    """
    计算细胞间基因表达的相似度，并构建稀疏矩阵。

    Parameters:
    -----------
    adata : AnnData
        包含基因表达数据的 AnnData 对象
    k_neighbors : int
        每个细胞选择最相似的邻居数量
    metric : str
        相似度度量方式，可选 'cosine', 'euclidean', 'correlation', 'pearson'

    Returns:
    --------
    KNN_df : pd.DataFrame
        稀疏矩阵，包含列 ['Cell1', 'Cell2', 'Distance']
    """

    # 将基因表达矩阵转为DataFrame，并设定细胞的index
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    # adata_Vars = adata
    X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)

    # 根据选择的度量方法计算相似度
    if metric == 'cosine':
        similarity_matrix = cosine_similarity(X)
    elif metric == 'euclidean':
        similarity_matrix = -euclidean_distances(X)  # 将距离转为相似度，距离越小相似度越大
    elif metric == 'pearson':
        # 使用皮尔逊相关系数
        similarity_matrix = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                similarity_matrix[i, j] = pearsonr(X.iloc[i, :], X.iloc[j, :])[0]
    else:
        raise ValueError(f"未知的相似度度量: {metric}")

    # 存储最相似的n_neighbors个细胞的关系
    KNN_list = []

    for i in range(similarity_matrix.shape[0]):
        # 获取与当前细胞最相似的n_neighbors个邻居（排除自身）
        sorted_indices = np.argsort(-similarity_matrix[i, :])  # 按相似度降序排序
        closest_cells = sorted_indices[1:k_neighbors + 1]  # 取最相似的n_neighbors个细胞（排除自己）
        closest_distances = similarity_matrix[i, closest_cells]  # 对应的相似度

        # 将每个细胞与其最相似的细胞对存入KNN_list
        KNN_list.append(pd.DataFrame({
            'Cell1': [i] * k_neighbors,
            'Cell2': closest_cells,
            'Distance': closest_distances
        }))

    # 将所有DataFrame合并
    KNN_df = pd.concat(KNN_list, ignore_index=True)

    # 将细胞编号映射回原来的index
    id_cell_trans = dict(zip(range(X.shape[0]), X.index))
    KNN_df['Cell1'] = KNN_df['Cell1'].map(id_cell_trans)
    KNN_df['Cell2'] = KNN_df['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (KNN_df.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (KNN_df.shape[0] / adata.n_obs))

    # 保存到 adata
    #adata.uns['Gene_Similarity_Net'] = KNN_df
    adata.uns['Gene_Similarity_Net_KNN'] = KNN_df

def Cal_Gene_Similarity_Net_Kmeans(adata, n_clusters=5, connect_type='intra_cluster_all', metric='euclidean'):
    """
    基于 K-Means 聚类构建细胞相似性图，支持欧氏距离和余弦相似度

    Parameters:
    -----------
    adata : AnnData
        输入的单细胞数据
    n_clusters : int
        聚类的簇数量
    connect_type : str
        连接类型，可选 'intra_cluster_all' 或 'intra_cluster_knn'
    metric : str
        距离度量，可选 'euclidean'（欧氏距离）或 'cosine'（余弦相似度）
    """
    # 数据准备
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    X = adata_Vars.X.toarray()

    # 执行 K-Means 聚类
    if metric == 'euclidean':
        # 使用 sklearn 的 KMeans（默认欧氏距离）
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        labels = kmeans.labels_
    elif metric == 'cosine':
        # 手动实现基于余弦相似度的 K-means
        n_samples, n_features = X.shape

        # 初始化质心（随机选择样本点）
        np.random.seed(0)
        centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]

        # K-means 迭代
        max_iter = 20
        for _ in range(max_iter):
            # 计算余弦相似度矩阵 (n_samples x n_clusters)
            sim_matrix = cosine_similarity(X, centroids)

            # 分配样本到最近的质心（注意：余弦相似度越高越相似）
            labels = np.argmax(sim_matrix, axis=1)

            # 更新质心
            new_centroids = np.zeros((n_clusters, n_features))
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    # 质心 = 簇内样本的均值（注意：此处仍使用欧氏空间的均值）
                    new_centroids[i] = np.mean(cluster_points, axis=0)

            # 检查收敛
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        labels = labels.astype(int)
    else:
        raise ValueError("metric 参数需为 'euclidean' 或 'cosine'")

    # 后续代码与原函数相同...
    adata.obs['kmeans_cluster'] = labels
    clusters = adata.obs['kmeans_cluster'].unique()
    kmeans_edges = []

    if connect_type == 'intra_cluster_all':
        cell_ids = adata.obs.index.tolist()
        for cluster in clusters:
            cluster_cells = adata.obs[adata.obs['kmeans_cluster'] == cluster].index.tolist()
            for i in range(len(cluster_cells)):
                for j in range(i + 1, len(cluster_cells)):
                    kmeans_edges.append({
                        'Cell1': cluster_cells[i],
                        'Cell2': cluster_cells[j],
                        'Distance': 1.0
                    })
    elif connect_type == 'intra_cluster_knn':
        for cluster in clusters:
            cluster_adata = adata[adata.obs['kmeans_cluster'] == cluster]
            cluster_knn_df = Cal_Gene_Similarity_Net(cluster_adata, k_neighbors=6, verbose=False)
            kmeans_edges.extend(cluster_knn_df.to_dict('records'))
    else:
        raise ValueError("连接类型需为 'intra_cluster_all' 或 'intra_cluster_knn'")

    kmeans_df = pd.DataFrame(kmeans_edges)
    adata.uns['Gene_Similarity_Net_KMeans'] = kmeans_df
    return kmeans_df

def get_intersection_KNN_KMeans(adata):
    """
    从 adata.uns 中取出 KNN 和 K-Means 图的边数据，计算并返回边交集
    """
    # 从 adata.uns 中获取两个图的数据
    knn_df = adata.uns.get('Gene_Similarity_Net_KNN')
    kmeans_df = adata.uns.get('Gene_Similarity_Net_KMeans')

    # 检查数据是否存在
    if knn_df is None or kmeans_df is None:
        raise ValueError("adata.uns 中缺少 Gene_Similarity_Net_KNN 或 Gene_Similarity_Net_KMeans 数据")

    # 定义函数，将边转为无序元组（统一格式，方便取交集）
    def edge_to_tuple(row):
        cell1, cell2 = row['Cell1'], row['Cell2']
        return tuple(sorted([cell1, cell2]))  # 小的 cell ID 在前，大的在后，保证无向边一致

    # 构建边集合
    knn_edges = set(knn_df.apply(edge_to_tuple, axis=1))
    kmeans_edges = set(kmeans_df.apply(edge_to_tuple, axis=1))

    # 计算交集
    common_edges = knn_edges & kmeans_edges

    # 从 knn_df 中筛选出交集的边（保留原始 Distance 等信息，也可用 kmeans_df，按需选）
    intersection_rows = []
    for edge in common_edges:
        # 匹配边（因为是无向边，两种顺序都要检查）
        mask = ((knn_df['Cell1'] == edge[0]) & (knn_df['Cell2'] == edge[1])) | \
               ((knn_df['Cell1'] == edge[1]) & (knn_df['Cell2'] == edge[0]))
        if not knn_df[mask].empty:
            intersection_rows.append(knn_df[mask].iloc[0].to_dict())

    # 转为 DataFrame
    intersection_df = pd.DataFrame(intersection_rows)

    # 也可以把交集结果存回 adata.uns，按需选择
    adata.uns['Gene_Similarity_Net_Intersection'] = intersection_df

    return intersection_df

def merge_local_global_net(adata, net_key1='Spatial_Net_local', net_key2='Gene_Similarity_Net_Intersection',
                   output_key='Spatial_Net', inplace=True):
    """
    合并两个无向图网络的边集，生成并集网络

    Parameters:
    -----------
    adata : AnnData
        包含网络数据的 AnnData 对象
    net_key1, net_key2 : str
        分别对应 adata.uns 中两个网络的键名
    output_key : str
        合并后的网络在 adata.uns 中存储的键名
    inplace : bool
        是否直接修改原 adata 对象，False 则返回新 DataFrame

    Returns:
    --------
    pd.DataFrame
        合并后的网络边集（仅当 inplace=False 时返回）
    """
    # 检查数据是否存在
    if net_key1 not in adata.uns or net_key2 not in adata.uns:
        missing_keys = [k for k in [net_key1, net_key2] if k not in adata.uns]
        raise ValueError(f"adata.uns 中缺少以下网络数据: {missing_keys}")

    # 获取两个网络数据
    network1 = adata.uns[net_key1]
    network2 = adata.uns[net_key2]

    # 定义函数将边转为无序元组（处理无向边）
    def edge_to_tuple(df_row):
        cell1, cell2 = df_row['Cell1'], df_row['Cell2']
        return tuple(sorted([cell1, cell2]))

    # 构建边集合（自动去重）
    edges_set1 = set(network1.apply(edge_to_tuple, axis=1))
    edges_set2 = set(network2.apply(edge_to_tuple, axis=1))

    # 计算并集
    union_edges_set = edges_set1.union(edges_set2)

    # 合并两个网络的所有边
    all_edges_df = pd.concat([network1, network2], ignore_index=True)

    # 筛选出并集边并去重
    all_edges_df['edge_tuple'] = all_edges_df.apply(edge_to_tuple, axis=1)
    union_df = all_edges_df[all_edges_df['edge_tuple'].isin(union_edges_set)].drop_duplicates(subset='edge_tuple')

    # 删除临时列
    union_df = union_df.drop(columns=['edge_tuple'])

    # 存储结果
    if inplace:
        adata.uns[output_key] = union_df
        print(f"已将并集网络存储到 adata.uns['{output_key}']")
        print(f"并集网络边数量：{len(union_df)}")
        return None
    else:
        return union_df

def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,
                       key_section='Section_id', section_order=None, verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff_2D
        radius cutoff for 2D SNN construction.
    rad_cutoff_Zaxis
        radius cutoff for 2D SNN construction for consturcting SNNs between adjacent sections.
    key_section
        The columns names of section_ID in adata.obs.
    section_order
        The order of sections. The SNNs between adjacent sections are constructed according to this order.
    
    Returns
    -------
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    num_section = np.unique(adata.obs[key_section]).shape[0]
    if verbose:
        print('Radius used for 2D SNN:', rad_cutoff_2D)
        print('Radius used for SNN between sections:', rad_cutoff_Zaxis)
    for temp_section in np.unique(adata.obs[key_section]):
        if verbose:
            print('------Calculating 2D SNN of section ', temp_section)
        temp_adata = adata[adata.obs[key_section] == temp_section, ]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_2D, verbose=False)
        temp_adata.uns['Spatial_Net']['SNN'] = temp_section
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0]/temp_adata.n_obs))
        adata.uns['Spatial_Net_2D'] = pd.concat(
            [adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])
    for it in range(num_section-1):
        section_1 = section_order[it]
        section_2 = section_order[it+1]
        if verbose:
            print('------Calculating SNN between adjacent section %s and %s.' %
                  (section_1, section_2))
        Z_Net_ID = section_1+'-'+section_2
        temp_adata = adata[adata.obs[key_section].isin(
            [section_1, section_2]), ]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
        spot_section_trans = dict(
            zip(temp_adata.obs.index, temp_adata.obs[key_section]))
        temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(
            spot_section_trans)
        temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(
            spot_section_trans)
        used_edge = temp_adata.uns['Spatial_Net'].apply(
            lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge, ]
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, [
            'Cell1', 'Cell2', 'Distance']]
        temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0]/temp_adata.n_obs))
        adata.uns['Spatial_Net_Zaxis'] = pd.concat(
            [adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])
    adata.uns['Spatial_Net'] = pd.concat(
        [adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
    if verbose:
        print('3D SNN contains %d edges, %d cells.' %
            (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %
            (adata.uns['Spatial_Net'].shape[0]/adata.n_obs))

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)


import numpy as np
from scipy.spatial import distance_matrix
from collections import Counter


def refine_label(adata, radius=50, key='label', new_key='refined_label'):
    """
    基于空间位置信息对细胞标签进行平滑处理

    参数:
    adata -- AnnData对象
    radius -- 考虑的邻居数量 (默认50)
    key -- 原始标签在adata.obs中的列名 (默认'label')
    new_key -- 新标签在adata.obs中的列名 (默认'refined_label')

    返回:
    修改后的AnnData对象
    """
    # 检查输入
    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found. Spatial coordinates are required.")

    if key not in adata.obs:
        raise ValueError(f"Key '{key}' not found in adata.obs.")

    # 获取空间坐标和原始标签
    position = adata.obsm['spatial']
    old_type = adata.obs[key].values

    # 计算距离矩阵
    distance = distance_matrix(position, position)

    n_cell = distance.shape[0]
    new_type = []

    # 对每个细胞进行标签优化
    for i in range(n_cell):
        # 获取当前细胞到所有细胞的距离
        dist_vec = distance[i, :]

        # 获取最近的radius个邻居的索引（不包括自身）
        nearest_indices = np.argsort(dist_vec)[1:radius + 1]

        # 获取邻居的标签
        neighbor_labels = old_type[nearest_indices]

        # 找到最常见的邻居标签
        if len(neighbor_labels) > 0:
            counter = Counter(neighbor_labels)
            most_common = counter.most_common(1)[0][0]
            new_type.append(most_common)
        else:
            # 如果没有邻居，保持原标签
            new_type.append(old_type[i])

    # 将新标签添加到adata.obs
    adata.obs[new_key] = np.array(new_type)

    # 添加函数参数到uns中记录
    adata.uns['refine_label_params'] = {
        'radius': radius,
        'original_key': key,
        'new_key': new_key
    }

    return adata

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SMML-Graph', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
