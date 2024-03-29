import numpy as np
from enum import Enum

def normalized_cm(cm):

    ncm = cm / np.sum(cm, axis=1).reshape(-1, 1)
    return ncm



# # from generalized_matth.matt_funct import  AVERAG_TYPE
# class AVERAG_TYPE(Enum):
#              HARMONIC = 0
#              ARITHMETIC = 1
#              GEOMETRIC= 2
#              LEHMER = 4
#
#
# class main_generalzied_object:
#     def __init__(self,average_val,p=0 ):
#
#         if average_val==1:
#             self.pair_mean = self.arithmetic_for_pairs
#             self.array_mean =   self.arithmetic_for_array
#         elif average_val == 0:
#             self.pair_mean = self.harmonic_for_pairs
#             self.array_mean = self.harmonic_for_array
#         elif average_val == 2:
#             self.pair_mean = self.geometric_for_pairs
#             self.array_mean = self.geometric_for_array
#         elif average_val == 4:
#             # self.pair_mean = self.lehmer_for_pairs
#             self.array_mean = self.lehmer_for_array
#             self.p = p
#
#         return
#
#     def check_array_positivity(self,x):
#         if np.min(x)<=0:
#            return 0
#         return 1
#
#     def check_positivity(self, x):
#         if x <= 0:
#             return 0
#         return 1
#
#     def arithmetic_for_array(self,x):
#         if self.check_array_positivity(x) :
#             return np.mean(x)
#         print ("Please provide positive data")
#         return -1
#     def arithmetic_for_pairs(self,x,y):
#         if self.check_positivity(x) and self.check_positivity(y) :
#             return 0.5*(x+y)
#         print ("Please provide positive data")
#         return -1
#     def harmonic_for_array(self,x):
#         if self.check_array_positivity(x) :
#             ss =np.sum([1/i for i in x])
#             return len(x)/ss
#         print ("Please provide positive data")
#         return -1
#     def harmonic_for_pairs(self,x,y):
#         if self.check_positivity(x) and self.check_positivity(y) :
#             return 2*(x*y)/(x+y)
#         print ("Please provide positive data")
#         return -1
#     def geometric_for_array(self,x):
#         if self.check_array_positivity(x) :
#             return np.exp(np.mean(np.log(x)))
#
#
#         print ("Please provide positive data")
#         return -1
#
#     def geometric_for_pairs(self, x, y):
#         if self.check_positivity(x) and self.check_positivity(y):
#             return np.sqrt(x * y)
#         print("Please provide positive data")
#         return -1
#     def lehmer_for_array(self, x):
#         if self.check_array_positivity(x):
#             return np.sum([xx**self.p for xx in x])/ np.sum([xx**(self.p-1) for xx in x])
#
#         print("Please provide positive data")
#         return -1
#

def pair_pfunc(array0, p):
        larr= len(array0)
        ss = np.sum([(np.poweri, p)  for i in array0])/larr
        s1 = np.power(ss, 1 / p)
        return s1


def build_socre(total_array,pval2):
    lt = len(total_array)
    if not(pval2  ==0):
        tot_arr2 = np.sum([np.power(i, pval2)   for i in total_array])/lt


        tot_arr_s = np.power(tot_arr2, 1 / pval2)
    else:
        tot_arr2 = np.exp(np.sum([np.log(i)   for i in total_array]) /lt )
    return tot_arr_s

def pair_wise_score(cm, pvalformena):
    ncm =normalized_cm(cm)
    total_array =[]
    lncm, _= ncm.shape
    for j in range (lncm-1):
        for i in range(j+1,lncm):
            total_array.append( build_socre([ncm[j,i],ncm[i,j]]  ,pvalformena))

    return total_array
def pair_wise_process(cm,pval1, pval2):
    total_array =  pair_wise_score(cm, pval1)
    score =build_socre(total_array,pval2)
    return score

