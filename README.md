# SMML-Graph

**Spatial Multi-omics Integration via Shared Spatial and Modality-specific Molecular Graph Learning**

SMML-Graph is a graph attention autoencoder framework for spatial multi-omics data analysis. It integrates multiple omics modalities (e.g., transcriptomics, proteomics, epigenomics) while preserving tissue spatial architecture, enabling accurate spatial domain identification.

## Overview

SMML-Graph addresses the challenge of integrating heterogeneous spatial omics data by:

- **Dual-graph construction**: A shared spatial neighbor graph (KNN on spatial coordinates) is combined with modality-specific expression similarity graphs (KNN ∩ K-Means intersection) to jointly capture spatial proximity and molecular similarity.
- **Dual-branch encoding & adaptive fusion**: Each modality is encoded by an independent graph attention encoder. A lightweight adaptive fusion module dynamically learns spot-specific fusion weights to generate unified joint representations.
- **Cross-modal contrastive learning**: A cosine embedding loss aligns embeddings from different modalities at the same spatial spot while enhancing discriminability across different spots.
- **Shared decoding**: A shared decoder reconstructs both modalities from the fused representation, ensuring cross-modal consistency.

Based on the PyTorch Geometric (pyG) framework, SMML-Graph is efficient and scalable for large-scale spatial multi-omics datasets.

## Supported Modalities

- Transcriptomics (RNA)
- Proteomics (Protein)
- Epigenomics (ATAC)
- Any combination of the above

## Software Dependencies

- scanpy
- pytorch
- pyG (PyTorch Geometric)
- scipy
- numpy
- pandas
- scikit-learn
