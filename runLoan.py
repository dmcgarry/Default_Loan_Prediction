#!/usr/local/python/bin/python

import os
import pandas as pd
import numpy as np
from itertools import chain
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score

vars = {
	'defaultRF'	:	['f777','f2','f527_diff_f528','f528_diff_f274','f222','logf271'],
	'defaultGBM'	:	['f777','f2','f527_diff_f528','f528_diff_f274','f222','f68'],
	'lossRF'	:	['f2','f528_diff_f274','f332','f67','f25','f120','f530','f766','f376','f670','f228','f652','f761','f406','f596','f777'],
	'lossGBM'	: 	['f2','f528_diff_f274','f332','f67','f25','f120','f766','f376','f39','f670','f228','f652','f415','f596','f406','f13','f355'],
	'lossSVM'	:	['f2','f332','f67','f25','f120','f766','f376','f670','f228','f652','f761','f4','f13','f386','f596','f9','f355','f406','f518','f328','f696','f674','f777','f718','f778_27']
}

models = {
	'defaultRF'	:	{'n_estimators':145,'max_depth':18, 'min_samples_split':4,'max_features':None},
	'defaultGBM'	:	{'n_estimators':65,'max_depth':6, 'learning_rate':0.300,'max_features':'sqrt'},
	'lossRF'	:	{'n_estimators':275,'max_depth':22, 'min_samples_split':2,'max_features':'sqrt'},
	'lossGBM'	:	{'n_estimators':160,'max_depth':5, 'learning_rate':0.109,'max_features':'sqrt'},
	'lossSVM'	:	{'C':2.5,'gamma': 0.027}
}
		
######################
## Helper Functions ##
######################
def mae(pred,obs):
	""" Mean absolute error """
	return np.mean(np.abs(pred-obs))

def bestF1(obs,pred):
	""" Find the optimal F1 value over a grid of cutoffs """
	best = 0
	bestcut = 0
	for cutoff in np.arange(0.01,0.99,0.01):
		tmp = f1_score(obs,pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
		if tmp > best:
			best = tmp
			bestcut = cutoff
	return best

def checkCutOff(loss,predLoss,predDefault):
	""" Find the best cutoff from the default model to the loss model """
	best = 99
	tmp = np.copy(predLoss)
	for cutoff in np.arange(0.01,0.99,0.01):
		tmp[predDefault < cutoff] = 0
		if mae(tmp,loss) < best:
			best = mae(tmp,loss)
			bestcut = cutoff
	return {'mae': best, 'Cutoff': bestcut}

def applyCut(lossPred,defaultPred,cut):
	""" Apply a cutoff from the default model to the loss model """
	pred = np.copy(lossPred)
	pred[defaultPred < cut] = 0.0
	return pred

####################
## Load/Prep Data ##
####################
def loadDat(wd='/usr/local/tmp/dmcgarry/kag/loan',vars=vars):
	""" Load train + test sets and prep data """
	#set working directory
	os.chdir(wd)
	#get unique vars
	uniqueVars = list(set(chain.from_iterable([x for x in vars.itervalues()])))
	#load training + test data
	train = pd.read_csv('train_v2.csv')	
	train['f778_27'] = train['f778'].apply(lambda x: 1 if x == 27 else 0)
	train['f527_diff_f528'] = train.f527 - train.f528
	train['f528_diff_f274'] = train.f528 - train.f274
	train['logf271'] = np.log(train['f271'] + 1)
	loss = train.loss
	train = train[uniqueVars]
	test = pd.read_csv('test_v2.csv')
	test['f778_27'] = test['f778'].apply(lambda x: 1 if x == 27 else 0)
	test['f527_diff_f528'] = test.f527 - test.f528
	test['f528_diff_f274'] = test.f528 - test.f274
	test['logf271'] =  np.log(test['f271'] + 1)
	id = test.id
	test = test[uniqueVars]
	#clean data
	for x in uniqueVars:
		if train[x].dtype not in ['float64']:
			train[x] = train[x].astype(float)
		if test[x].dtype != train[x].dtype:
			test[x] = test[x].astype(train[x].dtype)
		if not all(train[x].notnull()):
			train[x] = train[x].fillna(train[x].median())
		if not all(test[x].notnull()):
			test[x] = test[x].fillna(train[x].median())
	#scale data
	scale = StandardScaler()
	train = pd.DataFrame(scale.fit_transform(train))
	train.columns = uniqueVars
	train = train.join(loss)
	train['default'] = train.loss.apply(lambda x: 1 if x > 0 else 0)
	test = pd.DataFrame(scale.transform(test))
	test.columns = uniqueVars
	test = test.join(id)
	return train, test

###################
## Default Model ##
###################
def defaultModel(model,vars,t,train,test,seed):
	""" Make a model for default """
	cv = KFold(len(train),5,shuffle=True,random_state=seed)
	train[t+'DefaultPred'] = 0.0
	for tr, val in cv:
		model.fit(train[vars].ix[train.index[tr]],train['default'].ix[train.index[tr]])
		train[t+'DefaultPred'].ix[train.index[val]] = model.predict_proba(train[vars].ix[train.index[val]])[:,1]
	model.fit(train[vars],train.default)
	test[t+'DefaultPred'] = model.predict_proba(test[vars])[:,1]
	result = {'AUC':roc_auc_score(train.default, train[t+'DefaultPred']),'F1':bestF1(train.default,train[t+'DefaultPred'])}
	print t + " AUC: " + str(np.round(result['AUC'],5))
	print t + " F1:  " + str(np.round(result['F1'],5))
	return result

def runDefaultModels(train,test,seed):
	""" Run all default models """
	#RF Model
	rfDefault = defaultModel(
		RandomForestClassifier(n_estimators=models['defaultRF']['n_estimators'],max_depth=models['defaultRF']['max_depth'],min_samples_split=models['defaultRF']['min_samples_split'],max_features=models['defaultRF']['max_features'],n_jobs=10,random_state=seed+29),
		vars['defaultRF'],"rf",train,test,seed)
	#GBM Model
	gbmDefault = defaultModel(
		GradientBoostingClassifier(n_estimators=models['defaultGBM']['n_estimators'],learning_rate=models['defaultGBM']['learning_rate'],max_depth=models['defaultGBM']['max_depth'],max_features=models['defaultGBM']['max_features'],random_state=seed+29),
		vars['defaultGBM'],"gbm",train,test,seed)
	train['defaultPred'] = train['rfDefaultPred']*0.55 + train['gbmDefaultPred']*0.45
	test['defaultPred'] = test['rfDefaultPred']*0.55 + test['gbmDefaultPred']*0.45
	print "blended AUC: " + str(np.round(roc_auc_score(train.default, train['defaultPred']),5))
	print "blended F1:  " + str(np.round(bestF1(train.default,train['defaultPred']),5))
	return train,test

################
## Loss Model ##
################
def lossModel(model,vars,t,train,test,seed):
	""" Make a loss model """
	cv = KFold(len(train),5,shuffle=True,random_state=seed)
	train[t+'LossPred'] = 0.0
	for tr, val in cv:
		tmp = train.ix[train.index[tr]]
		tmp = tmp[tmp.default > 0]
		model.fit(tmp[vars],np.log(tmp.loss))
		train[t+'LossPred'].ix[train.index[val]] = np.e**model.predict(train[vars].ix[train.index[val]])
	tmp = train[train.default > 0]
	model.fit(tmp[vars],np.log(tmp['loss']))
	test[t+'LossPred'] = np.e**model.predict(test[vars])
	result = checkCutOff(train.loss,train[t+'LossPred'],train.defaultPred)
	print t + " MAE:", result['mae']
	return result

def runLossModels(train,test,seed):
	""" Run all loss models """
	#RF Model
	rfLoss = lossModel(
		RandomForestRegressor(n_estimators=models['lossRF']['n_estimators'],max_depth=models['lossRF']['max_depth'],min_samples_split=models['lossRF']['min_samples_split'],max_features=models['lossRF']['max_features'],n_jobs=10,random_state=seed+29),
		vars['lossRF'],"rf",train,test,seed)
	#GBM Model
	gbmLoss = lossModel(
		GradientBoostingRegressor(n_estimators=models['lossGBM']['n_estimators'],learning_rate=models['lossGBM']['learning_rate'],max_depth=models['lossGBM']['max_depth'],max_features=models['lossGBM']['max_features'],random_state=seed+29),
		vars['lossGBM'],"gbm",train,test,seed)
	#SVM Model
	svmLoss = lossModel(
		SVR(C=models['lossSVM']['C'],gamma=models['lossSVM']['gamma'],max_iter=10**5,random_state=seed+29),
		vars['lossSVM'],"svm",train,test,seed)
	#Blend Models
	minMAE = lambda x: mae(applyCut(train['rfLossPred'],train['defaultPred'],rfLoss['Cutoff'])*x[0] + applyCut(train['gbmLossPred'],train['defaultPred'],gbmLoss['Cutoff'])*x[1] + applyCut(train['svmLossPred'],train['defaultPred'],svmLoss['Cutoff'])*x[2],train.loss)
	weights = minimize(minMAE, [0.15,0.35,0.5], method='L-BFGS-B', bounds=((0,1),(0,1),(0,1)))
	print "blended MAE:",weights.fun
	test['loss'] = applyCut(test['rfLossPred'],test['defaultPred'],rfLoss['Cutoff'])*weights.x[0] + applyCut(test['gbmLossPred'],test['defaultPred'],gbmLoss['Cutoff'])*weights.x[1] + applyCut(test['svmLossPred'],test['defaultPred'],svmLoss['Cutoff'])*weights.x[2]
	return train, test
	
##########
## Main ##
##########
def main():
	""" Combine functions to make predictions """
	#load data
	train, test = loadDat()
	#make default models
	train, test = runDefaultModels(train, test, 5)
	#make loss models
	train, test = runLossModels(train, test, 5)
	#look at predictions
	print "Percent of Non Defaults:", test.loss.apply(lambda x: x > 0).mean()
	print "Average Loss:", test.loss.mean()
	#save prediction
	test[['id','loss']].to_csv("pred.csv",index=False)

# run everything when calling script from CLI
if __name__ == "__main__":
	main()
