# 导入相关数据包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
######################阴阳混在一起#######################
# train_test = pd.read_csv('oneDataFrameCleanKeepPatientsSelect2018PatientsLabledClean-DNAall.csv')
#######################阴/阳性###############################
train_test = pd.read_csv('oneDNAyang.csv')
seed = 123
np.random.seed(seed)
# print(train.head(5))
# print(train.info())
#print(train_test.describe())
# print(train['N-stage'].value_counts())
# train.groupby(['age'])['PFS'].mean().plot()
# sns.countplot('DNA-Binary',hue='PFS',data=train)
# plt.show()

#Feature Engineering
#将年龄划分是个阶段10以下,10-18,18-30,30-50,50以上
train_test['PFS'] = pd.cut(train_test['PFS'],right=False, bins=[0,24,1000],labels=[0,1])
print(train_test['PFS'].value_counts())
#对性别one-hot编码
train_test = pd.get_dummies(train_test,columns=["sex"])
#将年龄划分是个阶段10以下,10-18,18-30,30-50,50以上
train_test['age'] = pd.cut(train_test['age'], bins=[18,30,45,52,61,100],labels=[1,2,3,4,5])
train_test = pd.get_dummies(train_test,columns=['age'])
print(train_test.iloc[:,9:].head())

y = train_test['PFS']
print(y.value_counts())
# X = train_test[['sex_0','sex_1','age_1','age_2','age_3','age_4','age_5','hemoglobin','platelet']]
X = train_test.iloc[:,9:]
print(X.head())
# #不平衡处理（下采样）
# # #  分离多数和少数类别
# df_radio_semantic = train_test.iloc[:,13:-7]  #1725个特征
# df_y= train_test.iloc[:,2] #EGFR标签 1为突变
# df_concat = pd.concat([df_radio_semantic,df_y],axis=1,ignore_index=True)
# df_majority = df_concat[df_concat[4906]==0]
# df_minority = df_concat[df_concat[4906]==1]
# df_majority.reset_index(drop=True, inplace=True)
# df_minority.reset_index(drop=True, inplace=True)
# #无突变：EGFR突变 = 101：62
# df_majority_downsampled=df_majority.sample(frac=131/217,random_state=123)
# print('平衡处理后')
# rowlist=[]
# for indexs in df_majority_downsampled.index:
#     rowlist.append(indexs)
# print(rowlist)
# df_downsampled = pd.concat([df_majority_downsampled, df_minority])
# # # 分离输入特征 (X) 和目标变量 (y)
# y = df_downsampled[[4906]]
# X = df_downsampled.drop(4906, axis=1)
# X = np.array(X)


X = np.array(X)
y = np.ravel(np.array(y))

# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=seed,stratify=y)
# print('训练集大小：',X.shape)
# print('测试集大小：',X_test.shape)
# #归一化
# from sklearn.preprocessing import StandardScaler
# ss2 = StandardScaler()
# ss2.fit(X_train[['hemoglobin','platelet']])
# X_train[['hemoglobin','platelet']] = ss2.transform(X_train[['hemoglobin','platelet']])
# X_test[['hemoglobin','platelet']] = ss2.transform(X_test[['hemoglobin','platelet']])
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.ravel(np.array(y_train)).astype(np.int)
# y_test = np.ravel(np.array(y_test)).astype(np.int)

###############################特征选择##########################
# print('================开始特征选择==================')
# ###########################RFE
# # from sklearn.svm import LinearSVC
# # from sklearn.feature_selection import RFE
# # rfe = RFE(estimator=LinearSVC(), n_features_to_select=3)
# # rfe = rfe.fit(X_train, y_train)
# # X_train = rfe.transform(X_train)
# # X_test = rfe.transform(X_test)
# ###################树模型特征选择#########################
# # from sklearn.ensemble import ExtraTreesClassifier
# # from sklearn.datasets import load_iris
# # from sklearn.feature_selection import SelectFromModel
# #
# # clf = ExtraTreesClassifier()
# # clf = clf.fit(X_train, y_train)
# # model = SelectFromModel(clf, prefit=True)
# # X_train = model.transform(X_train)
# # X_test = model.transform(X_test)
#
# # #########################LASSO特征选择
# # from sklearn.svm import LinearSVC
# # from sklearn.datasets import load_iris
# # from sklearn.feature_selection import SelectFromModel
# #
# # # Load the boston dataset.
# # lsvc = LinearSVC(C=0.03, penalty="l1", dual=False).fit(X_train, y_train)
# # model = SelectFromModel(lsvc,prefit=True)
# # X_train = model.transform(X_train)
# # X_test = model.transform(X_test)
#
#
# ########################分类############################
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost.sklearn import XGBClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost.sklearn import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from mlxtend.classifier import StackingClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from mlxtend.classifier import StackingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn import svm
# from sklearn.ensemble import GradientBoostingClassifier
# rf = svm.SVC()
# rf.fit(X_train, y_train)
# #joblib.dump(rfe, 'rfe_fitted_13_EGFR_balanced_%s_cv%d.pkl'%(label,cv_index))
#
# y_test_pred = rf.predict(X_test)
# from sklearn.metrics import classification_report
# target_names = ['I期','II期','III期','IV期']
# print(classification_report(y_test, y_test_pred, target_names=target_names))
# print('准确率：',accuracy_score(y_test, y_test_pred))

####################交叉验证##########################
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

basemodel1 = RandomForestClassifier(oob_score=True, random_state=seed,class_weight='balanced')
basemodel2 = XGBClassifier(random_state=seed,class_weight='balanced')
basemodel3 = GaussianNB()
basemodel4 = LogisticRegression(random_state=seed,class_weight='balanced')
basemodel5 = svm.SVC(kernel='linear', probability=True, random_state=seed,class_weight='balanced')
basemodel6 = KNeighborsClassifier()
basemodel7 = AdaBoostClassifier(random_state=seed)
basemodel8 = CatBoostClassifier(random_state=seed,task_type="GPU")

lr = LogisticRegression()
# 使用逻辑回归进行模型融合
sclf = StackingClassifier(
    classifiers=[basemodel1, basemodel2, basemodel3, basemodel4, basemodel5, basemodel6, basemodel7,basemodel8],
    use_probas=True,
    average_probas=False,
    meta_classifier=lr)
j = 1
allaccuracy = []
allroc_auc = []
allSensitivity = []
allprecision = []
allSpecificity = []
plt.figure()
ind_savemodel = 0
# basemodel4 = svm.SVC(C=0.1, kernel='rbf', gamma=20, decision_function_shape='ovr')
for basemodel, label in zip(
        [basemodel1, basemodel2, basemodel3, basemodel4, basemodel5, basemodel6, basemodel7,basemodel8, sclf],
        [
            'Random Forest',
            'xgboost',
            'GaussianNB',
            'LogisticRegression',
            'SVM(linear)',
            'KNN',
            'AdaBoost',
            'CatBoost',
            'ModelStacking'
        ]):
    ##################画ROC曲线################
    plt.subplot(3, 3, j)
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from sklearn.externals import joblib
    # #############################################################################
    # Data IO and generation
    # Import some data to play with

    # #############################################################################
    # Classification and ROC analysis
    # Run classifier with cross-validation and plot ROC curves
    cv_fold = 5
    cv = StratifiedKFold(n_splits=cv_fold,random_state=123)
    classifier = basemodel
    tprs = []
    aucs = []
    TPR = []
    FPR = []

    tprr = []
    fprr = []
    accc = []
    ppvv = []
    acc_mean_xianai_test = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    cv_index = 1
    for train, test in cv.split(X, y):
        from sklearn.feature_selection import RFE
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        # #参数estimator为基模型
        #  #参数n_features_to_select为选择的特征个数
        # 平衡处理
        from imblearn.over_sampling import SMOTE, ADASYN
        print('平衡前', X[train].shape)
        xtrain, ytrain = SMOTE().fit_sample(X[train], y[train])

        print('平衡后', xtrain.shape)
        # 归一化
        from sklearn.preprocessing import StandardScaler

        ss2 = StandardScaler()
        ss2.fit(xtrain)
        xtrain = ss2.transform(xtrain)
        xtest = ss2.transform(X[test])
        xt = xtrain
        xtt = xtest

        # 特征选择
        # 选择测试集
        print('================开始特征选择==================')
        n_features = 40

        import matplotlib.pyplot as plt
        ###排序选择最优秀的特征
        # from sklearn.datasets import load_iris
        # from sklearn.feature_selection import SelectKBest
        # from sklearn.feature_selection import chi2  # 引入卡方检验统计量
        # # 对于回归: f_regression , mutual_info_regression
        # # 对于分类: chi2 , f_classif , mutual_info_classif
        # iris = load_iris()
        # X, y = iris.data, iris.target
        # print('源样本维度：',X.shape)
        #
        # X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
        # print('新样本维度：',X_new.shape)

        ###RFE
        # # 这里递归的移除最不重要的像素点来对每个像素点（特征）进行排序
        from sklearn.svm import LinearSVC
        from sklearn.datasets import load_digits
        from sklearn.feature_selection import RFE
        import matplotlib.pyplot as plt
        # 创建ref对象和每个像素点的重要度排名
        if ind_savemodel < cv_fold:
            rfe = RFE(estimator=LinearSVC(), n_features_to_select=n_features,step=20)
            rfe = rfe.fit(xt, ytrain)
            joblib.dump(rfe, './20190611实验/rfe_naso_only_positive_feat{}第%d次交叉验证模型.pkl'.format(n_features) % cv_index)

        if ind_savemodel > cv_fold-1:
            rfe = joblib.load('./20190611实验/rfe_naso_only_positive_feat{}第%d次交叉验证模型.pkl'.format(n_features) % cv_index)

        xt = rfe.transform(xt)
        xtt = rfe.transform(xtt)
        if ind_savemodel < cv_fold:
            print('第{}交叉验证特征选择后X_train_shape:'.format(cv_index), xt.shape)
            print('第{}交叉验证特征选择后X_val_shape:'.format(cv_index), xtt.shape)
        ind_savemodel += 1
        ########################################
        classifier.fit(xt, ytrain)
        if label == 'CatBoost':
            joblib.dump(classifier, './20190611实验/20190611_model/%s_fitted_{}feat_positive_cv%d.pkl'.format(n_features) %(label,cv_index))
        probas_ = classifier.predict_proba(xtt)
        y_val_pred = classifier.predict(xtt)



        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        from sklearn.metrics import confusion_matrix

        cnf_matrix = confusion_matrix(y[test], y_val_pred)
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
        i += 1
        cv_index += 1
        # print("gggg:",TPR)
        aucs.append(roc_auc)
        accc.append(ACC)
        ppvv.append(PPV)
        tprr.append(TPR)
        fprr.append(FPR)
    FPRR = np.mean(fprr)
    Specificity = 1 - FPRR
    print("[%s]测试集accuracy: %0.2f (+/- %0.2f) " % (label, np.mean(accc), np.std(accc, ddof=1)))
    print("[%s]测试集auc: %0.2f (+/- %0.2f) " % (label, np.mean(aucs), np.std(aucs, ddof=1)))
    print("[%s]测试集Precision: %0.2f (+/- %0.2f) " % (label, np.mean(ppvv), np.std(ppvv, ddof=1)))
    print("[%s]测试集Sensitivity: %0.2f (+/- %0.2f) " % (label, np.mean(tprr), np.std(tprr, ddof=1)))
    print("[%s]测试集Specificity: %0.2f (+/- %0.2f) " % (label, Specificity, np.std(fprr, ddof=1)))

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
    plt.title(label,loc='left',fontsize=10)
    plt.legend(loc="lower right", fontsize="xx-small")
    j = j + 1

    allaccuracy.append(np.mean(accc))
    allroc_auc.append(np.mean(aucs))
    allSensitivity.append(np.mean(tprr))
    allprecision.append(np.mean(ppvv))
    allSpecificity.append(np.mean(Specificity))
print(j)
plt.show()
#############################################################################
# 使用ggplot样式来模拟ggplot2风格的图形，ggplot2是一个常用的R语言绘图包
plt.figure()
plt.subplot(2, 3, 1)
from matplotlib import cm

label = ['Random Forest', 'xgboost', 'GaussNB', 'LogisticRegression',
         'SVM(linear)', 'KNN', 'AdaBoost','CatBoost', 'ModelStacking']
x = allaccuracy
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('accuracy')
plt.ylabel('Classifier')

plt.subplot(2, 3, 2)
from matplotlib import cm

x = allroc_auc
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('auc')
plt.ylabel('Classifier')

plt.subplot(2, 3, 3)
from matplotlib import cm

x = allSensitivity
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('Sensitivity')
plt.ylabel('Classifier')

plt.subplot(2, 3, 4)
from matplotlib import cm

x = allprecision
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('precision')
plt.ylabel('Classifier')

plt.subplot(2, 3, 5)
from matplotlib import cm

x = allSpecificity
idx = np.arange(len(x))
color = cm.jet(np.array(x) / max(x))
plt.barh(idx, x, color=color)
plt.yticks(idx + 0.4, label)
plt.grid(axis='x')
plt.xlabel('Specificity')
plt.ylabel('Classifier')
plt.show()
