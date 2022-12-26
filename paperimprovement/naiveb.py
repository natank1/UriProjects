# https://www.kaggle.com/code/errearanhas/iris-classification-using-xgboost
from  sklearn.datasets import make_classification
from generalized_matth.matt_funct import AVERAG_TYPE
from generalized_matth.matt_funct import matthew_multiclass
import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.naive_bayes import  MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from our_scores import  pair_wise_process

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
def pfunc(prec0,rec0,p):
    ss=0.5*(np.power(prec0,p)+np.power(rec0,p))
    s1 = np.power(ss,1/p)
    return s1


X, y = make_classification(
    n_samples=40000,  # row number
    n_features=20, # feature numbers
    n_informative=6, # The number of informative features
    n_classes = 4, # The number of classes
    random_state = 42 # random seed
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print (X_train.shape,X_test.shape)
from sklearn.pipeline import Pipeline
print ("Training Stage")
xgb = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])
xgb.fit(X_train,y_train)


outp =xgb.predict(X_test)
cm=   confusion_matrix(y_test, outp)
ncm =cm/np.sum(cm,axis=1).reshape(-1,1)
print (cm)
print (ncm)
pval1=2
pval2=-1
print (accuracy_score(y_test,outp))
y_true =y_test
y_pred =outp
print ("Gen Matthg")
test0 = matthew_multiclass(y_true, y_pred, avg_type=AVERAG_TYPE.MATTHEW_GEN.value)
print(test0.main_matthew_mult_class())
print ("genralized F1")
tets0= matthew_multiclass(y_true, y_pred, avg_type=AVERAG_TYPE.F1_GEN.value)
print(  tets0.main_matthew_mult_class())
print ("ggen score pairwise")
print (pair_wise_process(cm, pval1, pval2))
from plt_bar import plt_bar_o
plt_bar_o(cm)