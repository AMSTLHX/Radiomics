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
dataall = pd.read_csv('./EGFR_鳞癌腺癌_all表格.csv', engine = 'python')
dataall = pd.get_dummies(dataall,columns=["轮廓"])
dataall = pd.get_dummies(dataall,columns=["边界"])
alldata = dataall.dropna()
# print(alldata['EGFR突变状态'].value_counts())

X = alldata.iloc[:,2:]
X.iloc[:,-21] = X.iloc[:,-21].convert_objects(convert_numeric=True)
X = X.dropna()
print(X.head())
print(X['病理类型'].value_counts())
y = X['病理类型']
X = X.iloc[:,1:]
print(X.iloc[:,1692:].head())
# print(y.value_counts())
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
classifier = svm.SVC(class_weight='balanced')
tprs = []
aucs = []
TPR = []
FPR = []
tprr = []
fprr = []
accc = []
ppvv = []
Specificity = []
seed_selcted = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
# cv = StratifiedKFold(n_splits= 5, random_state=1000)
# for train, test in cv.split(X, y):

for k in range(1,11):
    print('\n', '================新的一次循环开始=================')
    seeds = random.randint(0, 1000)
    print('第{}次划分训练集测试集split_seed: {}'.format(k,seeds))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y, random_state=seeds)
    ###################显著性检验分析训练集测试集分布一致性################
    try:
        sextrain = X_train['性别'].value_counts()
        sextest = X_test['性别'].value_counts()
        sexa = sextrain[0]
        sexc = sextrain[1]
        sexb = sextest[0]
        sexd = sextest[1]
        _, p_sex = stats.fisher_exact([[sexa, sexb], [sexc, sexd]])

        agetrain = X_train['年龄'].describe()
        train_age_max = agetrain[7]
        train_age_min = agetrain[3]
        train_age_median = agetrain[5]
        agetest = X_test['年龄'].describe()
        test_age_max = agetest[7]
        test_age_min = agetest[3]
        test_age_median = agetest[5]
        _, p_age = stats.wilcoxon([train_age_max, train_age_min, train_age_median],
                                  [test_age_max, test_age_min, test_age_median],
                                  zero_method='wilcox', correction=False)

        train_stage = X_train['分级'].value_counts()
        test_stage = X_test['分级'].value_counts()
        _, p_stage = stats.wilcoxon([train_stage[1], train_stage[2], train_stage[3], train_stage[4]],
                                    [test_stage[1], test_stage[2], test_stage[3], test_stage[4]],
                                    zero_method='wilcox', correction=False)

        train_smoke = X_train['吸烟史'].value_counts()
        test_smoke = X_test['吸烟史'].value_counts()
        smokea = train_smoke[0]
        smokec = train_smoke[1]
        smokeb = test_smoke[0]
        smoked = test_smoke[1]
        _, p_smoke = stats.fisher_exact([[smokea, smokeb], [smokec, smoked]])

        MDtrain = X_train['MD'].describe()
        train_MD_max = MDtrain[7]
        train_MD_min = MDtrain[3]
        train_MD_median = MDtrain[5]
        MDtest = X_test['MD'].describe()
        test_MD_max = MDtest[7]
        test_MD_min = MDtest[3]
        test_MD_median = MDtest[5]
        _, p_MD = stats.wilcoxon([train_MD_max, train_MD_min, train_MD_median],
                                 [test_MD_max, test_MD_min, test_MD_median],
                                 zero_method='wilcox', correction=False)
    except KeyError:
        print('Key Error')
        continue
    alpha = 0.05
    if (p_sex < (alpha/2) or p_age < (alpha/2) or p_stage < (alpha/2) or p_smoke<(alpha/2) or p_MD < (alpha/2)):
        continue
    X_train = pd.get_dummies(X_train, columns=["分级"])
    X_test = pd.get_dummies(X_test, columns=["分级"])


    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    seed_selcted.append(seeds)

    radioFeat_train =X_train
    radioFeat_test = X_test
    ##################方差特征选择################
    from sklearn.feature_selection import VarianceThreshold  # 导入python的相关模块
    vad = VarianceThreshold(threshold=0.01)  # 表示剔除特征的方差大于阈值的特征Removing features with low variance
    radioFeat_train = vad.fit_transform(radioFeat_train)  # 返回的结果为选择的特征矩阵
    radioFeat_test = vad.transform(radioFeat_test)
    print('train_test_split_seed={} 方差选择radiomics特征个数为:{}'.format(seeds, radioFeat_train.shape[1]))

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
    ror = RobustScaler()
    radioFeat_train = ror.fit_transform(radioFeat_train)
    radioFeat_test = ror.transform(radioFeat_test)

#######################################################
    # from scipy.stats import pearsonr
    # from sklearn.feature_selection import SelectKBest, chi2
    # sel = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=1300).fit(xabstrain, y[train])
    # xptrain = sel.transform(xabstrain)
    # xptest = sel.transform(xabstest)
    # print("P特征选择后特征个数：", xptest.shape[1])
###########################################################
    lassocv = Lasso(alpha = 0.02,random_state=seeds)
    lasso = lassocv.fit(radioFeat_train, y_train)
    # print("选择的alpha值：",lasso.alpha_)
    mask = lasso.coef_ != 0
    radioFeat_train = radioFeat_train[:, mask]
    radioFeat_test = radioFeat_test[:, mask]
    print('train_test_split_seed={} LASSO选择特征个数为:{}'.format(seeds, radioFeat_train.shape[1]))
##################################RFE################

    rfe = RFE(estimator=LinearSVC(random_state=seeds), n_features_to_select=23)
    rfe = rfe.fit(radioFeat_train, y_train)
    radioFeat_train = rfe.transform(radioFeat_train)
    radioFeat_test = rfe.transform(radioFeat_test)
    print('train_test_split_seed={} RFE选择radio特征个数为:{}'.format(seeds, radioFeat_train.shape[1]))

    radio_and_cliSem_Feat_train = radioFeat_train
    radio_and_cliSem_Feat_test = radioFeat_test
    print(radio_and_cliSem_Feat_train.shape)
    from sklearn.feature_selection import RFECV
    # n_rfeFeatSelected = 12
    # rfe = RFE(estimator=LinearSVC(random_state=seeds), n_features_to_select=n_rfeFeatSelected)
    #
    # shapeFeat_train = rfe.fit_transform(shapeFeat_train, y_train)
    # shapeFeat_test = rfe.transform(shapeFeat_test)
    # print('train_test_split_seed={} RFE选择特征个数为:{}'.format(seeds, shapeFeat_train.shape[1]))

    ########################################
    params = [
        {'kernel': ['linear'], 'C': [0.0001,0.001,0.01,0.1,1, 10, 100,80,120]},
        {'kernel': ['poly'], 'C': [0.0001,0.001,0.01,0.1,1,10,100,80,120], 'degree': [1,2, 3]},
        {'kernel': ['rbf'], 'C': [0.0001,0.001,0.01,0.1,1,10,100,80,120], 'gamma': [10,1, 0.1, 0.01, 0.001]}
    ]
    model = GridSearchCV(svm.SVC(probability=True,random_state=seeds),
                            params,
                            refit=True,
                            return_train_score=True,  # 后续版本需要指定True才有score方法
                            cv=5,
                    scoring= 'roc_auc' )
    # model.fit(shapeFeat_train, y_train)
    model.fit(radio_and_cliSem_Feat_train, y_train)

    print(model.best_params_)
    classifier = model.best_estimator_
    # probas_ = classifier.predict_proba(shapeFeat_test)
    # y_val_pred = classifier.predict(shapeFeat_test)
    probas_ = classifier.predict_proba(radio_and_cliSem_Feat_test)
    y_val_pred = classifier.predict(radio_and_cliSem_Feat_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC split_seed=%d (AUC = %0.2f)' % (seeds, roc_auc))
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(y_test, y_val_pred)
    # print(cnf_matrix.shape)
    # print("混淆矩阵:", cnf_matrix)

    FP = cnf_matrix[0, 1]
    FN = cnf_matrix[1, 0]
    TP = cnf_matrix[1, 1]
    TN = cnf_matrix[0, 0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    SPECIFICITY = 1-FPR
    i += 1
    # print("gggg:",TPR)
    aucs.append(roc_auc)
    accc.append(ACC)
    ppvv.append(PPV)
    tprr.append(TPR)
    Specificity.append(SPECIFICITY)
    print(roc_auc)
    print()

print("测试集mean_accuracy: %0.2f (+/- %0.2f) " % (np.mean(accc), np.std(accc, ddof=1)))
print("测试集mean_auc: %0.2f (+/- %0.2f) " % (np.mean(aucs), np.std(aucs, ddof=1)))
print("测试集mean_Precision: %0.2f (+/- %0.2f) " % (np.mean(ppvv), np.std(ppvv, ddof=1)))
print("测试集mean_Sensitivity: %0.2f (+/- %0.2f) " % (np.mean(tprr), np.std(tprr, ddof=1)))
print("测试集mean_Specificity: %0.2f (+/- %0.2f) " % (np.mean(Specificity), np.std(Specificity, ddof=1)))
print('划分训练集测试集split_seed: {}'.format(seed_selcted))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('svm',loc='left',fontsize=15)
plt.legend(loc="lower right", fontsize="xx-small")
j = j + 1
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
plt.subplot(2, 3, 1)
from matplotlib import cm

label = seed_selcted
x = accc
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color,tick_label=seed_selcted)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('accuracy')
plt.ylabel('split_seed')

plt.subplot(2, 3, 2)
from matplotlib import cm

x = aucs
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('auc')
plt.ylabel('split_seed')

plt.subplot(2, 3, 3)
from matplotlib import cm

x = tprr
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('Sensitivity')
plt.ylabel('split_seed')

plt.subplot(2, 3, 4)
from matplotlib import cm

x = ppvv
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('precision')
plt.ylabel('split_seed')

plt.subplot(2, 3, 5)
from matplotlib import cm

x = Specificity
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('Specificity')
plt.ylabel('split_seed')
plt.show()
