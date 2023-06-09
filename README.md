# PLPCA
Robust Persistent Laplacian Supervised Discriminative Sparse Principal Component Analysis

This work builds upon previous improvements to PCA, most notably the gLPCA procedure of Jiang et. al, the SDSPCA procedure of Feng et. al, and the RLSDSPCA procedure of Zhang et. al, by introducing the Persistent Laplacian. The Persistent Laplacian is induced via a filtration procedure on the Weighted Graph Laplacian, which is analogous to the Vietoris Rips complex. This term then captures local geometrical structure information through Persistent Spectral Graphs. 

---

![model](./PLPCA.pdf)

---

## MODEL:

Much of the code provided has been sourced from the previous work of Zhang et. al at the RLSDSPCA GitHub Repository referenced in their paper. 

Included here are several files for different PCA methods that can be used to compare relative classification performances on different datasets (after dimensionality reduction). Included are: PCA, gLPCA, gLSPCA, SDSPCA, RLSDSPCA, pLPCA, and PLPCA, which we also may refer to as RPLSDSPCA. 

## DATASETS: 

All datasets were obtained from the Cancer Genome Atlas database (https://portal.gdc.cancer.gov/). The MultiSource dataset is a four class dataset containing gene expression data and three cancer types: cholangiocarcinoma (CHOL), head and neck squamous cell carcinoma (HNSCC), and pancreatic adenocarcinoma (PAAD), as well as normal samples. The COAD dataset is a two class dataset containing gene expression data, colon adenocarcinoma samples, and normal tissue samples. Both datasets were used in this study as the benchmark datasets on which we performed our analysis. 

## REFERENCES: 

I.) C.-M. Feng, Y. Xu, J.-X. Liu, Y.-L. Gao, and C.-H. Zheng, “Supervised discriminative sparse PCA for com-characteristic gene selection and tumor classification on multiview biological data,” IEEE transactions on neural networks and learning systems, vol. 30, no. 10, pp. 2926-2937, 2019.

II.) B. Jiang, C. Ding, and J. Tang, "Graph-Laplacian PCA: Closed-form solution and robustness." pp. 3492-3498.

III.) Lu-Xing Zhang, He Yan, Yan Liu, Jian Xu, Jiangning Song, and Dong-Jun Yu. Enhancing characteristic gene selection and tumor classification by the robust laplacian supervised discriminative sparse pca. Journal of Chemical Information and Modeling, 62(7):1794–1807, 2022.

IV.) R. Wang, D. D. Nguyen, and G.-W. Wei. Persistent spectral graph. International Journal for Numerical Methods in Biomedical Engineering, page e3376, 2020.

## CITING:

You may use the following bibtex entry to cite RPLSDSPCA:

...

## INSTALLATION: 

You can install PLPCA directly from the repo in your terminal via: 

```bash
git clone https://github.com/seanfcottrell/PLPCA.git
```

