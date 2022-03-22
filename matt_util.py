import pandas as pd
import numpy as np
import scipy.stats as sc

gm =sc.mstats.gmean


def genralize_mat(mat):
    m1 = mat / mat.sum(axis=1)
    m2 = mat / mat.sum()
    G_mat = [[gm([m1.loc[i][j], m2.loc[j][i]]) for i in m1.index] for j in m1.columns]
    return np.linalg.det(G_mat)


if __name__ =='__main__':
    a = [[5, 6, 2], [2, 8, 11], [8, 2, 10]]
    mat = pd.DataFrame(data=a)
    print("genral_val=",genralize_mat(mat))
    print("For Indeity " ,genralize_mat(pd.DataFrame(data=np.identity(4))))