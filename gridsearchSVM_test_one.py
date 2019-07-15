# 导入相关数据包


from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import warnings
import pandas as pd
from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from scipy.stats import randint as sp_randint
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import random
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.linear_model import LassoCV,Lasso
import scipy.stats as stats
from collections import Counter
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")
dataall = pd.read_csv('../EGFR_only腺癌_delete异常.csv', engine = 'python')
dataall = pd.get_dummies(dataall,columns=["轮廓"])
dataall = pd.get_dummies(dataall,columns=["边界"])
alldata = dataall.dropna()
# print(alldata['EGFR突变状态'].value_counts())
X = alldata.iloc[:,1:]
X.iloc[:,-21] = X.iloc[:,-21].convert_objects(convert_numeric=True)
X = X.dropna()
y = X['EGFR突变状态']
X = X.iloc[:,1:]
print(y.value_counts())
# X = np.array(X)
# y = np.ravel(np.array(y))


def man(xinput,yinput,k):
    cc = []
    i = xinput.shape[1]
    for m in range(0,i):
        u_statistic, pVal = stats.mannwhitneyu(xinput[:,m],yinput)
        if pVal < k:
            cc.append(True)
        else:
            cc.append(False)

    return cc





seeda = 123
j = 1
allaccuracy = []
allroc_auc = []
allSensitivity = []
allprecision = []
allSpecificity = []
plt.figure()

seed = []
plt.subplot(1, 1, j)
classifier = svm.SVC()
tprs = []
aucs = []
TPR = []
FPR = []
tprs_train = []
aucs_train = []
TPR_train = []
FPR_train = []
tprr = []
fprr = []
accc = []
ppvv = []
Specificity = []
seed_selcted = []
mean_fpr = np.linspace(0, 1, 100)
mean_fpr_train = np.linspace(0, 1, 100)

i = 0
# cv = StratifiedKFold(n_splits= 5, random_state=1000)
# for train, test in cv.split(X, y):
seeds = 910
print('划分训练集测试集split_seed: {}'.format(seeds))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=seeds)
###################显著性检验分析训练集测试集分布一致性################
# sextrain = X_train['性别'].value_counts()
# sextest = X_test['性别'].value_counts()
# sexa = sextrain[0]
# sexc = sextrain[1]
# sexb = sextest[0]
# sexd = sextest[1]
# _, p_sex = stats.fisher_exact([[sexa, sexb], [sexc, sexd]])
#
# agetrain = X_train['年龄'].describe()
# train_age_max = agetrain[7]
# train_age_min = agetrain[3]
# train_age_median = agetrain[5]
# agetest = X_test['年龄'].describe()
# test_age_max = agetest[7]
# test_age_min = agetest[3]
# test_age_median = agetest[5]
# _, p_age = stats.wilcoxon([train_age_max, train_age_min, train_age_median],
#                           [test_age_max, test_age_min, test_age_median],
#                           zero_method='wilcox', correction=False)
#
# train_stage = X_train['分级'].value_counts()
# test_stage = X_test['分级'].value_counts()
# _, p_stage = stats.wilcoxon([train_stage[1], train_stage[2], train_stage[3], train_stage[4]],
#                             [test_stage[1], test_stage[2], test_stage[3], test_stage[4]],
#                             zero_method='wilcox', correction=False)
#
# train_smoke = X_train['吸烟史'].value_counts()
# test_smoke = X_test['吸烟史'].value_counts()
# smokea = train_smoke[0]
# smokec = train_smoke[1]
# smokeb = test_smoke[0]
# smoked = test_smoke[1]
# _, p_smoke = stats.fisher_exact([[smokea, smokeb], [smokec, smoked]])
#
# MDtrain = X_train['MD'].describe()
# train_MD_max = agetrain[7]
# train_MD_min = agetrain[3]
# train_MD_median = agetrain[5]
# MDtest = X_test['年龄'].describe()
# test_MD_max = agetest[7]
# test_MD_min = agetest[3]
# test_MD_median = agetest[5]
# _, p_MD = stats.wilcoxon([train_MD_max, train_MD_min, train_MD_median],
#                          [test_MD_max, test_MD_min, test_MD_median],
#                          zero_method='wilcox', correction=False)
# alpha = 0.05
# if (p_sex < (alpha/2) or p_age < (alpha/2) or p_stage < (alpha/2) or p_smoke<(alpha/2) or p_MD < (alpha/2)):
#     continue

seed_selcted.append(seeds)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 使用SMOTE方法进行过抽样处理
# model_smote = SMOTE(random_state=123)  # 建立SMOTE模型对象
# xsmot, ysmot = model_smote.fit_sample(X[train], y[train])
# print("原始特征个数：",xsmot.shape[1])
#######重抽样前的类别比例
# print('采样前1的个数：',np.sum(y_train))
# print('采样前1:0的个数:',np.sum(y_train) / len(y_train))
# # 重抽样后的类别比例
# print('采样后1的个数：', np.sum(yt))
# print('采样后1:0的个数:',np.sum(yt)/ len(yt))
# from imblearn.over_sampling import ADASYN
# print('平衡前', X[train].shape)
# Xt, yt = ADASYN().fit_sample(X[train], y[train])
print('------------------开始特征选择---------------------')
print('原始特征个数为:{}'.format(X_train.shape[1]))
##################方差特征选择################
from sklearn.feature_selection import VarianceThreshold  # 导入python的相关模块
sel = VarianceThreshold(threshold=0.01)  # 表示剔除特征的方差大于阈值的特征Removing features with low variance
ss = sel.fit(X_train)  # 返回的结果为选择的特征矩阵
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)
print('train_test_split_seed={} 方差选择特征个数为:{}'.format(seeds, X_train.shape[1]))

######################特征归一化到【-1，1】之间#####################
# max_abs_scaler = preprocessing.MaxAbsScaler()
# max_abs_scaler.fit(xmantrain)
# xabstrain = max_abs_scaler.transform(xmantrain)
# xabstest = max_abs_scaler.transform(xmantest)
##################方差特征选择################
# from sklearn.feature_selection import VarianceThreshold  # 导入python的相关模块
# sel = VarianceThreshold(threshold=0.01)  # 表示剔除特征的方差大于阈值的特征Removing features with low variance
# ss = sel.fit(xmantrain)  # 返回的结果为选择的特征矩阵
# xvartrain = sel.transform(xmantrain)
# xvartest = sel.transform(xmantest)
# print("方差特征选择后特征个数：", xvartest.shape[1])
####################标准化######################
from sklearn.preprocessing import StandardScaler, RobustScaler
#robust
ss3 = RobustScaler()
ss3.fit(X_train)
X_train = ss3.transform(X_train)
X_test = ss3.transform(X_test)
#######################################################
# from scipy.stats import pearsonr
# from sklearn.feature_selection import SelectKBest, chi2
# sel = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=1300).fit(xabstrain, y[train])
# xptrain = sel.transform(xabstrain)
# xptest = sel.transform(xabstest)
# print("P特征选择后特征个数：", xptest.shape[1])
###########################################################
lassocv = Lasso(alpha = 0.02,random_state=seeds)
lasso = lassocv.fit(X_train, y_train)
# print("选择的alpha值：",lasso.alpha_)
mask = lasso.coef_ != 0
X_train = X_train[:, mask]
X_test = X_test[:, mask]
print('train_test_split_seed={} LASSO选择特征个数为:{}'.format(seeds, X_train.shape[1]))
##################################RFE################

rfe = RFE(estimator=LinearSVC(random_state=seeds), n_features_to_select=25)
rfe = rfe.fit(X_train, y_train)
X_train = rfe.transform(X_train)
X_test = rfe.transform(X_test)
print('train_test_split_seed={} RFE选择特征个数为:{}'.format(seeds, X_train.shape[1]))

########################################
params = [
        {'kernel': ['linear'], 'C': [0.0001,0.001,0.01,0.1,1, 10, 100]},
        {'kernel': ['poly'], 'C': [0.0001,0.001,0.01,0.1,1,10,100], 'degree': [1,2, 3]},
        {'kernel': ['rbf'], 'C': [0.0001,0.001,0.01,0.1,1,10,100], 'gamma': [10,1, 0.1, 0.01, 0.001]}
    ]
# model = GridSearchCV(svm.SVC(probability=True),
#                         params,
#                         refit=True,
#                         return_train_score=True,  # 后续版本需要指定True才有score方法
#                         cv=5,
#                 scoring= 'roc_auc' )

model = svm.SVC(probability=True,C=100,kernel='linear',random_state=seeds)
model.fit(X_train, y_train)
# print(model.best_params_)
classifier = model
probas_ = classifier.predict_proba(X_test)
y_val_pred = classifier.predict(X_test)
probas_train = classifier.predict_proba(X_train)
y_train_pred = classifier.predict(X_train)


fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, probas_train[:, 1])
tprs.append(interp(mean_fpr, fpr, tpr))
tprs_train.append(interp(mean_fpr_train, fpr_train, tpr_train))
tprs[-1][0] = 0.0
tprs_train[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
roc_auc_train = auc(fpr_train, tpr_train)
plt.plot(fpr_train, tpr_train, lw=2, alpha=0.5, color='b',
         label='ROC-Train (AUC = %0.2f)' % (roc_auc_train))
plt.plot(fpr, tpr, lw=2, alpha=0.5,color='r',
         label='ROC-Test (AUC = %0.2f)' % (roc_auc))

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test, y_val_pred)
cnf_matrix_train = confusion_matrix(y_train, y_train_pred)
# print(cnf_matrix.shape)
# print("混淆矩阵:", cnf_matrix)

FP = cnf_matrix[0, 1]
FN = cnf_matrix[1, 0]
TP = cnf_matrix[1, 1]
TN = cnf_matrix[0, 0]
FP_train = cnf_matrix_train[0, 1]
FN_train = cnf_matrix_train[1, 0]
TP_train = cnf_matrix_train[1, 1]
TN_train = cnf_matrix_train[0, 0]
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
TPR_train = TP_train / (TP_train + FN_train)
# Specificity or true negative rate
TNR = TN / (TN + FP)
TNR_train = TN_train / (TN_train + FP_train)
# Precision or positive predictive value
PPV = TP / (TP + FP)
PPV_train = TP_train / (TP_train + FP_train)
# Negative predictive value
NPV = TN / (TN + FN)
NPV_train = TN_train / (TN_train + FN_train)
# Fall out or false positive rate
FPR = FP / (FP + TN)
FPR_train = FP_train / (FP_train + TN_train)
# False negative rate
FNR = FN / (TP + FN)
FNR_train = FN_train / (TP_train + FN_train)
# False discovery rate
FDR = FP / (TP + FP)
FDR_train = FP_train / (TP_train + FP_train)
# Overall accuracy
ACC = (TP + TN) / (TP + FP + FN + TN)
ACC_train = (TP_train + TN_train) / (TP_train + FP_train + FN_train + TN_train)
i += 1

# # print("gggg:",TPR)
# aucs.append(roc_auc)
# accc.append(ACC)
# ppvv.append(PPV)
# tprr.append(TPR)
# fprr.append(FPR)

Specificity = 1 - np.array(FPR)
Specificity_train = 1 - np.array(FPR_train)
label = 'SVM'
print("[%s]训练集accuracy: %0.2f " % (label, ACC_train))
print("[%s]训练集auc: %0.2f" % (label, roc_auc_train))
print("[%s]训练集Precision: %0.2f" % (label, PPV_train))
print("[%s]训练集Sensitivity: %0.2f" % (label, TPR_train))
print("[%s]训练集Specificity: %0.2f" % (label, Specificity_train))
print('------------------------------------------------------')
print("[%s]测试集accuracy: %0.2f " % (label, ACC))
print("[%s]测试集auc: %0.2f" % (label, roc_auc))
print("[%s]测试集Precision: %0.2f" % (label, PPV))
print("[%s]测试集Sensitivity: %0.2f" % (label, TPR))
print("[%s]测试集Specificity: %0.2f" % (label, Specificity))
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k',
                     label='Chance', alpha=.3)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title(label,loc='left',fontsize=10)
plt.legend(loc="lower right", fontsize="xx-small")
# aucs.append(roc_auc)
# accc.append(ACC)
# ppvv.append(PPV)
# tprr.append(TPR)
# Specificity.append(SPECIFICITY)
# allaccuracy.append(np.mean(accc))
# allroc_auc.append(np.mean(aucs))
# allSensitivity.append(np.mean(tprr))
# allprecision.append(np.mean(ppvv))
# allSpecificity.append(np.mean(Specificity))
plt.show()
#############################################################################
# 使用ggplot样式来模拟ggplot2风格的图形，ggplot2是一个常用的R语言绘图包
plt.figure()
plt.subplot(1,1,1)
from matplotlib import cm
# print("[%s]测试集accuracy: %0.2f " % (label, ACC))
# print("[%s]测试集auc: %0.2f" % (label, roc_auc))
# print("[%s]测试集Precision: %0.2f" % (label, PPV))
# print("[%s]测试集Sensitivity: %0.2f" % (label, TPR))
# print("[%s]测试集Specificity: %0.2f" % (label, Specificity))
label = ['acc','auc','precision','recall','specificity']
x = [ACC,roc_auc, PPV, TPR, Specificity]
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color,tick_label=label)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('Performance')
plt.ylabel('Evaluation Indicator ')
plt.show()
