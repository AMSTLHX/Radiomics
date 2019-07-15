import pandas as pd
import os, csv
import numpy as np
import shutil

allpath = os.getcwd()#用于返回当前工作目录。
# clinicalFileDir = allpath + '重新整理表格.xlsx'
dframe = pd.read_excel('重新整理表格.xlsx')
patientID = dframe['pid']
# print(patientID)
FeaturePidOne = pd.read_csv('F:\\PythonCodes\\BreastCancer\\001\\radiomicsFeatures.csv',header=None)
columns = list(dframe.columns)+list(FeaturePidOne.iloc[:,0])

print(columns)

sortedDataFrame = pd.DataFrame(columns = columns)
row_index = 0
path =  os.listdir(allpath)

for i in patientID:
    filenames = [z for z in path if z == str(i).zfill(3)]
    sortedDataFrame.loc[row_index, 'pid'] = i
    dfT2 = pd.read_excel(allpath + '\\重新整理表格.xlsx')
    for r in dfT2.columns:
        sortedDataFrame.loc[row_index, r] = dfT2.loc[row_index,r]
    if filenames != []:
        filename = filenames[0]
        filename = allpath + '\\%s\\radiomicsFeatures.csv' % filename
        print(filename)
        df = pd.read_csv(filename, engine='python', header=None)
        ind = 0
        for r in list(df.iloc[:,0]):
            sortedDataFrame.loc[row_index,  r] = df.at[ind,1]
            ind += 1
        else:
            pass
    row_index = row_index + 1
sortedDataFrame.to_csv('allFeatures.csv',encoding='gb18030')


# for i in patientID:
#     n = str(i)
#     m = "%03d" % i
#     filenames = [z for z in path if str(z) == str(m)]
#     sortedDataFrame.loc[row_index, 'pid'] = m
#     dfT2 = pd.read_excel(allpath + '\\重新整理表格.xlsx')
#     for r in dfT2.columns:
#         sortedDataFrame.loc[row_index, r] = dfT2.loc[row_index,r]
#     if filenames != []:
#         dataFolder = os.path.abspath(os.path.join(allpath,str(m)))
#         f = open(dataFolder+'\\radiomicsFeatures.csv')
#         print(f)
#         e = pd.read_csv(f,engine='python')
#         ee = np.transpose(e)
#         for r in ee:
#             sortedDataFrame.loc[m, r] =ee[r].values[0]
#     else:
#         pass
#     row_index = row_index + 1
# sortedDataFrame.to_csv('allRadiomicsFeaturesImageRescaled.csv',encoding='gb18030')
