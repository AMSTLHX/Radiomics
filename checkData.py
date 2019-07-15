# -*- coding: utf-8 -*-
import pandas as pd
import os, csv
import numpy as np
import shutil

path = os.getcwd()#用于返回当前工作目录。
# os.path.abspath()返回一个目录的绝对路径,os.path.join()函数：连接两个或更多的路径名组件

dataFolder = os.path.abspath(os.path.join(path,"..\\..\\..")) + '\\data\\'

clinicalFileDir = dataFolder + 'allTable\\clinicalTable-selectedPatients.csv'
radiomicsFeatureNameFile = dataFolder + 'allTable\\featureName.csv'
radiomicsFeatureNameColumns = pd.read_csv(radiomicsFeatureNameFile, index_col = 0).columns

radiomicsFolderNoRescaled = dataFolder + 'allTable\\imageNoRescaledTable\\'
radiomicsFolderRescaled = dataFolder + 'allTable\\imageRescaledTable\\'
clinicalDataFrame = pd.read_csv(clinicalFileDir, index_col = 0)
patientID = clinicalDataFrame['PatientID']

radiomicsFilesNoRescaled = os.listdir(radiomicsFolderNoRescaled)
radiomicsFilesRescaled = os.listdir(radiomicsFolderRescaled)

radiomicsFeatureNameColumnsCET1 = ['CET1-' + i for i in radiomicsFeatureNameColumns]
radiomicsFeatureNameColumnsT2 = ['T2-' + i for i in radiomicsFeatureNameColumns]
columns = list(clinicalDataFrame.columns) + list(radiomicsFeatureNameColumnsCET1) + list(radiomicsFeatureNameColumnsT2)

#Rescaled-1024-002aiyiming-1-radiomicsFeatures
#002aiyiming-1-radiomicsFeatures

sortedDataFrame = pd.DataFrame(columns = columns)


for i in patientID:
	
	filenames = [z for z in radiomicsFilesRescaled if z[14:17] == str(i).zfill(3)]
	sortedDataFrame.loc[row_index, 'PatientID'] = i

	if filenames != []:
		filenameCET1 = filenames[0]
		filenameT2 = filenames[1]
		dfCET1 = pd.read_csv(radiomicsFolderRescaled + filenameCET1, index_col = 0)
		dfT2 = pd.read_csv(radiomicsFolderRescaled + filenameT2, index_col = 0)

		for r in dfCET1.columns[1:]:
			sortedDataFrame.loc[row_index, 'CET1-' + r] = dfCET1[r].values[0]
		for r in dfT2.columns[1:]:
			sortedDataFrame.loc[row_index, 'T2-' + r] = dfT2[r].values[0]
	else:
		pass
	row_index = row_index + 1


sortedDataFrame.to_csv('allRadiomicsFeaturesImageRescaled.csv')






"""
featureColumns = pd.read_csv(path + '\\featureName.csv').columns
columns = list(clinicalDataFrame.columns) + list(featureColumns)

filenames = os.listdir(patients_dir)
for f in filenames:
	shutil.copy(patients_dir + f, path + '\\radiomicsFeatureTables\\table-1\\' + f[0:3]+'-'+f[-23:-22]+'.csv')




def checkMissingTable(path, string):
"""
