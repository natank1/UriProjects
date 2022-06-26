import pandas as pd
import numpy as np
# import scipy as sc
from scipy import stats as st


class matthew_mult:
    def __init__(self):
        self.hm = st.mstats.hmean
        self.gm = st.mstats.gmean
        return
    def genralize_mat(self, mat_obj):
        m1 = mat_obj / mat_obj.sum(axis=1)
        m2 = mat_obj / mat_obj.sum()
        G_mat = [[self.gm([m1.loc[i][j], m2.loc[j][i]]) for i in m1.index] for j in m1.columns]
        return np.linalg.det(G_mat)

    def ratio_mat(self,mat_obj, average):
        m1 = mat_obj / mat_obj.sum(axis=1)
        m2 = mat_obj / mat_obj.sum()

        return [[average([m1.loc[i][j], m2.loc[j][i]])
                 for i in m1.index] for j in m1.columns]


    def genralize_f1(self,mat_obj ):
        h_mat = self.ratio_mat(mat_obj, self.hm)
        return self.hm([h_mat[i][i] for i in range(len(mat_obj))])



    def genralize_mat(self, mat_obj):
        g_mat = self.ratio_mat(mat_obj, self.gm)
        return np.linalg.det(g_mat)



if __name__ =='__main__':
    a = [[5, 6, 2], [2, 8, 11], [8, 2, 10]]
    mat = pd.DataFrame(data=a)

    mattt0= matthew_mult()
    print("hm=",mattt0.hm)
    print("For Indeity " ,mattt0.genralize_mat(pd.DataFrame(data=np.identity(4))))
    print("Gen f1=",mattt0.genralize_f1(mat))
    print("gen F1 iDENT=",mattt0.genralize_f1(pd.DataFrame(data=
                                    np.identity(4))))
    print("worked on")
