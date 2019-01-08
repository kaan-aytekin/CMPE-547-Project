import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import warnings
from pgmpy.models import NaiveBayes
from pgmpy.models import BayesianModel
from random import random,sample,seed
from sklearn import metrics
warnings.simplefilter("ignore")
with open("data.pkl", "rb") as fp:
	data=pickle.load(fp)

#Delete products with sale volume less than 150
data=data.loc[:,data.sum()>150]
#Delete products that are not actually products (check,nan)
data=data.loc[:, data.columns.difference(["check","nan"])]
try:
	with open("Edges.txt", "rb") as fp:
		Edges=pickle.load(fp)
	with open("Nodes.txt", "rb") as fp:
		Nodes=pickle.load(fp)
	with open("CPD.txt", "rb") as fp:
		CPD=pickle.load(fp)
	with open("RandomColumns.txt", "rb") as fp:
		random_columns=pickle.load(fp)
	with open("RandomIndices.txt", "rb") as fp:
		random_indices=pickle.load(fp)
	data=data.iloc[:,random_columns]
	column_size=data.shape[1]
	#Delete invoices with all zeros from the data
	data=data[(data.T != 0).any()]
	row_size=data.shape[0]
	smallDF=data.iloc[random_indices,:]
	smallDF.shape
	DictOfModels={}
	for productName in smallDF.columns:
		print('Collecting model for {0}'.format(productName))
		model = NaiveBayes()
		model.add_nodes_from(Nodes[productName])
		model.add_edges_from(Edges[productName])
		model.add_cpds(*CPD[productName])
		DictOfModels[productName]=model
		#Save edge ,node, CPD information
	PseudoCounts={}
	#Pseudocounts are given (1,1) for uniform
	for productName in smallDF.columns:
	    PseudoCounts[productName]=[1,1]
except:
	print('Existing model not found')
	#Select random invoice (2000) and products (50)
	seed(0)
	column_size=data.shape[1]
	random_columns=sample(range(column_size),100)
	data=data.iloc[:,random_columns]
	#Delete invoices with all zeros from the data
	data=data[(data.T != 0).any()]
	row_size=data.shape[0]
	random_indices=sample(range(row_size),2000)
	smallDF=data.iloc[random_indices,:]
	smallDF.shape
	PseudoCounts={}
	#Pseudocounts are given (1,1) for uniform
	for productName in smallDF.columns:
	    PseudoCounts[productName]=[1,1]
	DictOfModels={}
	Edges={}
	Nodes={}
	CPD={}
	for productName in smallDF.columns:
		print('Building model for {0}'.format(productName))
		model = NaiveBayes()
		model.fit(smallDF,productName)
		DictOfModels[productName]=model
		#Save edge ,node, CPD information
		Edges[productName]=model.edges()
		Nodes[productName]=model.nodes()
		CPD[productName]=model.get_cpds()
	with open("Edges.txt", "wb") as fp:
		pickle.dump(Edges, fp)

	with open("Nodes.txt", "wb") as fp:
		pickle.dump(Nodes, fp)

	with open("CPD.txt", "wb") as fp:
		pickle.dump(CPD, fp)

	with open("RandomColumns.txt", "wb") as fp:
		pickle.dump(random_columns, fp)

	with open("RandomIndices.txt", "wb") as fp:
		pickle.dump(random_indices, fp)
	ProductNames=smallDF.columns
	with open("ProductNames.txt", "wb") as fp:
		pickle.dump(ProductNames, fp)

try:
	with open("testSetIndicies.txt", "rb") as fp:
		testSetIndicies=pickle.load(fp)
except:
	testSetIndicies=sample(set(range(row_size))-set(random_indices),500)
	with open("testSetIndicies.txt", "wb") as fp:
		pickle.dump(testSetIndicies, fp)	

testDF=data.iloc[testSetIndicies,:]
TrueValuesAndPredictions=[]
for index in range(testDF.shape[0]):
	print('Making prediction for invoice {0}/{1}'.format(index,testDF.shape[0]))
	testInstance=testDF.copy()
	testInstanceRow=testInstance.iloc[index,:]
	boughtItemsIndex=[i for i,x in enumerate(testInstanceRow==1) if x==True]
	unboughtItemsIndex=[i for i,x in enumerate(testInstanceRow==1) if x==False]
	#50% of the time predict for bought items 50% of the time predict for the unbought items
	if random()<0.5:
		itemToPredict=sample(boughtItemsIndex,1)
		DroppedColumnName=testInstanceRow.index[itemToPredict]
		testingInstanceDropped=testInstance.drop(DroppedColumnName,axis=1, inplace=False)
		predictedProbability=DictOfModels[DroppedColumnName[0]].predict_probability(testingInstanceDropped.iloc[index].to_frame().transpose()).iloc[0,1]
		#predictedProbability=model.predict_probability(testingInstanceDropped.iloc[index].to_frame().transpose()).iloc[0,1]
		TrueValue=testInstance.iloc[index,itemToPredict].iloc[0]
		TrueValuesAndPredictions.append((TrueValue,predictedProbability))
	else:
		itemToPredict=sample(unboughtItemsIndex,1)
		DroppedColumnName=testInstanceRow.index[itemToPredict]
		testingInstanceDropped=testInstance.drop(DroppedColumnName,axis=1, inplace=False)
		predictedProbability=DictOfModels[DroppedColumnName[0]].predict_probability(testingInstanceDropped.iloc[index].to_frame().transpose()).iloc[0,1]
		#predictedProbability=model.predict_probability(testingInstanceDropped.iloc[index].to_frame().transpose()).iloc[0,1]
		TrueValue=testInstance.iloc[index,itemToPredict].iloc[0]
		TrueValuesAndPredictions.append((TrueValue,predictedProbability))

#DRAW ROC CURVE AND CALCULATE AUC
TrueValuesAndPredictionsSeparated=list(zip(*TrueValuesAndPredictions))
trueVals=TrueValuesAndPredictionsSeparated[0]
preds=TrueValuesAndPredictionsSeparated[1]
fpr, tpr, threshold = metrics.roc_curve(trueVals, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(num='ROC CURVE')
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
input("Press any key to close the program")