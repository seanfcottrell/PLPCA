## RPLSDSPCA
Robust Persistent Laplacian Discriminative Sparse Principal Component Analysis

This work builds upon previous improvements to PCA, most notably the RLSDSPCA procedure of Zhang et. al, by introducing a Persistent Laplacian via a filtration procedure on the Weighted Graph Laplacian. This term then captures local geometrical structure information of high dimensional data via Laplacian Eigenmaps, and performs better due to its ability to emphasize topologically persistent features.

## MODEL:

Included here are several files for different PCA methods that can be used to compare relative classification performances on different datasets (after dimensionality reduction). Included are: PCA, gLPCA, gLSPCA, SDSPCA, RLSDSPCA, RPLSDSPCA. 

## DATASETS: 

All datasets were obtained from the Cancer Genome Atlas database (). The MultiSource dataset is a four class dataset containing gene expression data and three cancer types: cholangiocarcinoma (CHOL), head and neck squamous cell carcinoma (HNSCC), and pancreatic adenocarcinoma (PAAD), as well as normal samples. The COAD dataset is a two class dataset containing gene expression data, colon adenocarcinoma samples, and normal tissue samples. Both datasets were used in this study as the benchmark datasets on which we performed our analysis. 

## REFERENCES: 

I.) C.-M. Feng, Y. Xu, J.-X. Liu, Y.-L. Gao, and C.-H. Zheng, “Supervised discriminative sparse PCA for com-characteristic gene selection and tumor classification on multiview biological data,” IEEE transactions on neural networks and learning systems, vol. 30, no. 10, pp. 2926-2937, 2019.

II.) B. Jiang, C. Ding, and J. Tang, "Graph-Laplacian PCA: Closed-form solution and robustness." pp. 3492-3498.

III.) Lu-Xing Zhang, He Yan, Yan Liu, Jian Xu, Jiangning Song, and Dong-Jun Yu. Enhancing characteristic gene selection and tumor classification by the robust laplacian supervised discriminative sparse pca. Journal of Chemical Information and Modeling, 62(7):1794–1807, 2022.

IV.) R. Wang, D. D. Nguyen, and G.-W. Wei. Persistent spectral graph. International Journal for Numerical Methods in Biomedical Engineering, page e3376, 2020.

## CITING:

You may use the following bibtex entry to cite RPLSDSPCA:

...

## INSTALLATION: 

You can install RPLSDSPCA directly from the repo via: 

```bash
git clone ...
```

