# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 14:55:32 2014

"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split,StratifiedKFold,cross_val_score
import sklearn.preprocessing as pp
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import metrics
from collections import defaultdict
from sklearn import svm
from sklearn import tree
import string
import operator
import nltk

path = '/desktop/kaggle/pizza/train.json'
test_path = '/desktop/kaggle/pizza/test.json'
stop_path = '/desktop/stopwords.txt'

o = open(stop_path)
lines = o.readlines()
stop_words = []

for line in lines:
    stop_words.append(line.strip('\n'))

pizza = pd.read_json(path)
test_init = pd.read_json(test_path)

pizza.info()
#
#target = path['requester_recieved_pizza']

def remv_punc(sting):
    for c in string.punctuation:
        sting = sting.replace(c, '')
    return sting
       
def make_dicts(lst):
    """
    returns a sorted (by num of occurences) list of tuples (word, number of occurences)
    as well as a dict {word: number of occurences}
    """
    
    word_dict = defaultdict(int)
    bigram_dict = defaultdict(int)
    for request in lst:
        rqst = []
        for word in remv_punc(request.lower()).split():
            if word not in stop_words:        
                rqst.append(word)
                word_dict[word]+=1
        rqst_big = list(nltk.bigrams(rqst))
        for big in rqst_big:
            bigram_dict[big]+=1
    return sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True), word_dict,sorted(bigram_dict.items(), key=operator.itemgetter(1), reverse=True), bigram_dict

def ctr(request, lst):
    ctr = 0
    for word in remv_punc(request.lower()).split():
        if word in lst:
            ctr +=1
    return ctr
    
def big_ctr(request,lst):
    ctr = 0
    rqst = []
    for word in remv_punc(request.lower()).split():
        if word not in stop_words:
            rqst.append(word)
    rqst_big = list(nltk.bigrams(rqst))
    for bigram in rqst_big:
        if bigram in lst:
            ctr +=1
    return ctr


def make_pcent_good(good_words, good_dict, bad_words, bad_dict):
    pcent = []
    """
    returns a list of tuples and a score dict with key/values:
    (word, ratio of times word appears in recieved requests vs.
    denied (weighted by population size))
    """

    for key in good_words:
        if (key[1])>5:
            nish = []
            nish.append(key[0])
            nish.append('Good: '+str(key[1])+' Bad: '+str(bad_dict[key[0]]))
            nish.append(good_dict[key[0]]/(max(1.0*bad_dict[key[0]], .60))-.4)            
            pcent.append(nish)
    
    sort_p = sorted(pcent, key=operator.itemgetter(2), reverse=True)
        
    for word in sort_p:
        word.append(word[2])
    
    score_dict = defaultdict(int)

    for word in sort_p:
        score_dict[word[0]]=word[-1]

    return sort_p, score_dict
    

def remove_retrieval_fields(train_init):
    """
    takes dataframe
    removes fields that are not in the test set
    returns dataframe
    """
    train_init = train_init.drop(['request_text','requester_user_flair','number_of_downvotes_of_request_at_retrieval','number_of_upvotes_of_request_at_retrieval','post_was_edited','request_number_of_comments_at_retrieval','requester_account_age_in_days_at_retrieval','requester_days_since_first_post_on_raop_at_retrieval','requester_number_of_comments_at_retrieval','requester_number_of_comments_in_raop_at_retrieval','requester_number_of_posts_at_retrieval','requester_number_of_posts_on_raop_at_retrieval','requester_upvotes_minus_downvotes_at_retrieval','requester_upvotes_plus_downvotes_at_retrieval'], axis=1)
    return train_init

def word_score(arry,score_dict):
    score = 0    
    for word in remv_punc(arry.lower()).split():
        if word not in stop_words:
            score+=score_dict[word]
    return score

def big_score(arry, score_dict):
    score = 0
    rqst = []
    for word in remv_punc(arry.lower()).split():
        if word not in stop_words:
            rqst.append(word)
    rqst_big = list(nltk.bigrams(rqst))
    for bigram in rqst_big:
        score += score_dict[bigram]
    return score

#removes fields that are not in test set
pizza = remove_retrieval_fields(pizza)     

#divides training set into those who recieved pizza and those 
#who did not
accepted_requests = pizza[pizza['requester_received_pizza']==True]
rejected_requests = pizza[pizza['requester_received_pizza']==False]

#creates arrays of accepted and rejected texts 
accepted_texts = accepted_requests['request_text_edit_aware']
rejected_texts = rejected_requests['request_text_edit_aware']

#converts the arrays to lists
a_t = accepted_texts.tolist()
r_t = rejected_texts.tolist()

#creates a list of good/bad words, a dict of how often these words occur
#as well as similar objects for bigrams
good_words, good_dict, good_big, big_dict_g = (make_dicts(a_t))
bad_words, bad_dict, bad_big, big_dict_b = (make_dicts(r_t))

    
sorted_pcent, score_dict = make_pcent_good(good_words, good_dict, bad_words, bad_dict)
sort_bigrams, big_dict = make_pcent_good(good_big, big_dict_g, bad_big, big_dict_b)

good_words_t50 = [x[0] for x in sorted_pcent[:50]]
bad_words_t50 = [x[0] for x in sorted_pcent[-50:]]

good_bigs_t100 = [x[0] for x in sort_bigrams[:100]]
bad_bigs_t100 = [x[0] for x in sort_bigrams[-100:]]

pizza['num_good_words'] = pizza.apply(lambda x: ctr(x['request_text_edit_aware'], good_words_t50), axis = 1)
pizza['num_bad_words'] = pizza.apply(lambda x: ctr(x['request_text_edit_aware'], bad_words_t50), axis = 1)
pizza['good_minus_bad_words'] = pizza.apply(lambda x: x['num_good_words']-x['num_bad_words'], axis =1)
pizza['request_score'] = pizza.apply(lambda x: word_score(x['request_text_edit_aware'],score_dict), axis=1)

pizza['num_good_bigs'] = pizza.apply(lambda x: big_ctr(x['request_text_edit_aware'],good_bigs_t100), axis = 1)
pizza['num_bad_bigs'] = pizza.apply(lambda x: big_ctr(x['request_text_edit_aware'],bad_bigs_t100), axis = 1)
pizza['good_minus_bad_bigs'] = pizza.apply(lambda x: x['num_good_bigs']-x['num_bad_bigs'], axis =1)
pizza['request_score_bigs'] = pizza.apply(lambda x: big_score(x['request_text_edit_aware'],big_dict), axis=1)


test_init['num_good_words'] = test_init.apply(lambda x: ctr(x['request_text_edit_aware'], good_words_t50), axis = 1)
test_init['num_bad_words'] = test_init.apply(lambda x: ctr(x['request_text_edit_aware'], bad_words_t50), axis = 1)
test_init['good_minus_bad_words'] = test_init.apply(lambda x: x['num_good_words']-x['num_bad_words'], axis =1)
test_init['request_score'] = test_init.apply(lambda x: word_score(x['request_text_edit_aware'],score_dict), axis=1)

test_init['num_good_bigs'] = test_init.apply(lambda x: big_ctr(x['request_text_edit_aware'],good_bigs_t100), axis = 1)
test_init['num_bad_bigs'] = test_init.apply(lambda x: big_ctr(x['request_text_edit_aware'],bad_bigs_t100), axis = 1)
test_init['good_minus_bad_bigs'] = test_init.apply(lambda x: x['num_good_bigs']-x['num_bad_bigs'], axis =1)
test_init['request_score_bigs'] = test_init.apply(lambda x: big_score(x['request_text_edit_aware'],big_dict), axis=1)

accepted_subs = accepted_requests['request_title']
rejected_subs = rejected_requests['request_title']

a_s = accepted_subs.tolist()
r_s = rejected_subs.tolist()

good_words, good_dict, good_big, big_dict_g = (make_dicts(a_s))
bad_words, bad_dict, bad_big, big_dict_b = (make_dicts(r_s))
    
sorted_pcent, sub_dict = make_pcent_good(good_words, good_dict, bad_words, bad_dict)

good_lab_t15 = [x[0] for x in sorted_pcent[:15]]
bad_lab_t30 = [x[0] for x in sorted_pcent[-30:]]

pizza['label_good_words'] = pizza.apply(lambda x: ctr(x['request_title'], good_lab_t15), axis = 1)
pizza['label_bad_words'] = pizza.apply(lambda x: ctr(x['request_title'], bad_lab_t30), axis = 1)
pizza['good_minus_bad_labels'] = pizza.apply(lambda x: x['label_good_words']-x['label_bad_words'], axis =1)
pizza['request_score_sub'] = pizza.apply(lambda x: word_score(x['request_title'],sub_dict), axis=1)

test_init['label_good_words'] = test_init.apply(lambda x: ctr(x['request_title'], good_lab_t15), axis = 1)
test_init['label_bad_words'] = test_init.apply(lambda x: ctr(x['request_title'], bad_lab_t30), axis = 1)
test_init['good_minus_bad_labels'] = test_init.apply(lambda x: x['label_good_words']-x['label_bad_words'], axis =1)
test_init['request_score_sub'] = test_init.apply(lambda x: word_score(x['request_title'],sub_dict), axis=1)

labels_init = pizza['requester_received_pizza']

train_init = pizza.drop(['requester_received_pizza'], axis=1)

def create_features(df):
    df['giver_exists']=df.apply(lambda x: x['giver_username_if_known'] != 'N/A', axis=1)
    df['request_length']=df['request_text_edit_aware'].map(lambda x: len(x.split()))
    return df

def binning(df):
    bins = pd.qcut(train_init['request_length'], 8,retbins=True)[-1]
    df['request_bins'] = pd.cut(df['request_length'],bins)
    bins2 = pd.qcut(train_init['requester_upvotes_minus_downvotes_at_request'].tolist(),8,retbins=True)[-1]    
    df['up_minus_down_bins'] = pd.cut(df['requester_upvotes_minus_downvotes_at_request'].tolist(),bins2)
    return df, bins, bins2

train_init = create_features(train_init)
test_init = create_features(test_init)
pizza = create_features(pizza)

train_init, bins, bins2 = binning(train_init)

test_init['request_bins'] = pd.cut(test_init['request_length'], bins)
test_init['up_minus_down_bins'] = pd.cut(test_init['requester_upvotes_minus_downvotes_at_request'].tolist(), bins2)

pizza['request_bins'] = pd.cut(pizza['request_length'], bins)
pizza['up_minus_down_bins'] = pd.cut(pizza['requester_upvotes_minus_downvotes_at_request'].tolist(), bins2)

def drop_obj_fields(df):
    return df.drop(['request_id','giver_username_if_known','request_title','request_text_edit_aware','requester_subreddits_at_request','requester_username'],axis=1)

test_ids = test_init['request_id']

train_init = drop_obj_fields(train_init)
test_init = drop_obj_fields(test_init)
pizza = drop_obj_fields(pizza)
56
#pizza.to_csv('C:/users/brjohn/desktop/pizza.csv')
exp_test = test_init
exp_test['requester_received_pizza'] = exp_test.apply(lambda x: '?', axis =1)
exp_test.to_csv('C:/users/brjohn/desktop/pizza_test3.csv')

def make_dummies(df, field):
    dummies = pd.get_dummies(df[field], prefix=field[:-2])
    df = df.drop(field, axis=1)
    return df.join(dummies)

train_init = make_dummies(train_init,'request_bins')
train_init = make_dummies(train_init,'up_minus_down_bins')

test_init = make_dummies(test_init,'request_bins')
test_init = make_dummies(test_init,'up_minus_down_bins')

#Two methods for splitting the data into train and test sets
def dsplit(train_init,labels_init,tsize):
    train, test, train_target, test_target=train_test_split(train_init, labels_init, test_size=tsize, random_state=42)
    return train,test,train_target,test_target

def dsplit2(train_init,labels_init,tsize):
    msk = np.random.rand(len(train_init)) < tsize
    train = train_init[msk]
    train_target = labels_init[msk]
    test = train_init[~msk]
    test_target = train_init[~msk]
    return train, test, train_target, test_target
    

def crossvalid(train_init,labels_init,folds):
    skf = StratifiedKFold(labels_init, n_folds=folds)
    
    for train_index, test_index in skf:
        train = train_init.ix[train_index.tolist()]
        train_target = labels_init.ix[train_index.tolist()]
        test = train_init.ix[test_index.tolist()]
        test_target = labels_init.ix[test_index.tolist()]
        
    return test, test_target, train, train_target
    
#Normalize data
def normalize(train, test,test_init):
    norm = pp.Normalizer()
    train = norm.fit_transform(pd.DataFrame(train,dtype=float))    
    test = norm.transform(pd.DataFrame(test,dtype=float))
    test_final = norm.transform(pd.DataFrame(test_init,dtype=float))
    return train, test,test_final

def MinMax(train, test, test_init, whole_train):
    minmax = pp.MinMaxScaler()
    train = minmax.fit_transform(train)
    test = minmax.transform(test)
    train_final = minmax.transform(whole_train)
    test_final = minmax.transform(test_init)
    return train, test, test_final, train_final
    
def scaler(train,test,test_init):
    scaler = pp.StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    test_final = scaler.transform(test_init)
    return train, test, test_final

#Pull out important features
def pca(train, train_target, test, test_target, test_init, n):
    pca = PCA(n_components = n, whiten = True)
    train = pca.fit_transform(train, train_target)
    test = pca.transform(test)
    test_final = pca.transform(test_init)
    return train, test, test_final
        
#build models

def knnclassifier(train, train_target, test, test_target, k):
    classif = KNeighborsClassifier(n_neighbors = k,algorithm='kd_tree',weights='uniform',p=1)
    classif.fit(train,train_target)
    res = classif.predict(train)
    
    print '*************************** knn ****************'
    print classification_report(train_target,res)
    
    res1 = classif.predict(test)
    
    print classification_report(test_target, res1)
    return classif

def gbr(train,test,train_target,test_target, loss_func='ls', lr=.1, n_est=100, m_d=3):
    clf = GradientBoostingRegressor(loss=loss_func, learning_rate=lr, n_estimators=n_est, max_depth=m_d)
    clf.fit(train, train_target)
    res = clf.predict(train)
    
    print 'MAE'
    print metrics.mean_absolute_error(train_target,res)
    print 'MSE'    
    print metrics.mean_squared_error(train_target,res)
    
    res1 = clf.predict(test)
    print 'MAE'
    print metrics.mean_absolute_error(test_target,res1)
    print 'MSE'    
    print metrics.mean_squared_error(test_target,res1)
    return clf

def gbc(train,test,train_target,test_target, lr=.1, n_est=100):
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=lr, n_estimators=n_est)
    clf.fit(train, train_target)
    res = clf.predict(train)
    
    print '*************************** GBC ****************'
    print classification_report(train_target,res)
    
    res1 = clf.predict(test)
    print classification_report(test_target, res1)
    return clf
    
def AdaBC(train,test,train_target,test_target,weights=None, n=500, lr = 1):
    abc = AdaBoostClassifier(n_estimators = n, learning_rate = lr)
    abc.fit(train, train_target, sample_weight = weights)
    res = abc.predict(train)
    
    print '*************************** AdaBC ****************'
    print classification_report(train_target,res)
    
    res1 = abc.predict(test)
    print classification_report(test_target,res1)
    return abc

def svm_class(train, test, train_target,test_target,C=1.0):
    clf = svm.LinearSVC()
    clf.fit(train, train_target)
    res = clf.predict(train)
    
    print classification_report(train_target,res)
    
    res1 = clf.predict(test)
    print classification_report(test_target, res1)
    return clf

def rand_forest(train,test,train_target,test_target,weights=None, n=10, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, rand_state = 42):
    clf = RandomForestClassifier(n_estimators = n)
    clf.fit(train, train_target, sample_weight=weights)
    res = clf.predict(train)
    
    print '*************************** rand_forest ****************'
    print classification_report(train_target,res)
    
    res1 = clf.predict(test)
    print classification_report(test_target, res1)
    return clf

def Dec_tree(train, test, train_target,test_target):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train, train_target)
    res = clf.predict(train)
    
    print '*************************** Decision Tree ****************'
    print classification_report(train_target,res)
    
    res1 = clf.predict(test)
    print classification_report(test_target, res1)
    return clf

def class_tester(classif, train, test, train_target, test_target):
    clf=classif
    clf.fit(train,train_target)
    res = clf.predict(train)
    
    print classification_report(train_target,res)
    
    res1 = clf.predict(test)
    print classification_report(test_target, res1)

train,test,train_target,test_target = dsplit(train_init,labels_init,.10)

#train,test,test_final = scaler(train,test,test_init)
#train,test,test_final = pca(train,train_target,test,test_target, test_final,4)

#train,test,test_final = normalize(train,test,test_init)
#train, test, test_final = pca(train, train_target, test, test_target, test_final, 4)

train,test,test_final,train_final= MinMax(train,test, test_init, train_init)
#train, test, test_final = pca(train, train_target, test, test_target, test_final, 10)
ann_train = pd.DataFrame(train).join(pd.DataFrame(train_target,columns=['class']))
ann_test = pd.DataFrame(test).join(pd.DataFrame(test_target,columns=['class']))

#train, test, test_final = pca(train, train_target, test, test_target, test_init, 4)

for i in range(1,11):
    print str(i)+'**********************************'
    knnclassifier(train, train_target, test, test_target, i)

knnclassifier(train, train_target, test, test_target, 5)
AdaBC(train,test,train_target,test_target)
Dec_tree(train,test,train_target,test_target)
rand = rand_forest(train,test,train_target,test_target,n=50)
drak = gbc(train,test,train_target,test_target)
linSVC = svm_class(train,test,train_target,test_target)
#gbr(train,train_target,test,test_target,5)

#Run program: split data, apply model
    
def make_submission_file(clf):
    clf.fit(train_final,labels_init)
    guesses = clf.predict(test_final)
    final_guess = pd.DataFrame(guesses,columns=['requester_received_pizza'],dtype=int).join(test_ids)
    final_guess.set_index('request_id', inplace=True)
    final_guess.to_csv('/desktop/submission.csv')


#ANN with pybrain
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer

ds = ClassificationDataSet(41,1)
for i in range(len(train_final)):
    indata = tuple(train_final[i])
    outdata = labels_init[i]
    ds.addSample(indata,outdata)

ts = ClassificationDataSet(41,1)
for i in range(len(test)):
    indata = tuple(test[i])
    outdata = test_target[i]
    ds.addSample(indata,outdata)

#ft = pybrain.datasets.ClassificationDataSet(37,1)
#for i in range(len(test_final)):
#    indata = tuple(test[i])
#    ft.addSample(indata)

ds._convertToOneOfMany()
ts._convertToOneOfMany()

n = buildNetwork(ds.indim,14,14,ds.outdim,recurrent=True,outclass=SoftmaxLayer)
t = BackpropTrainer(n, dataset=ds, learningrate=0.01,momentum=0.5,verbose=True)
t.trainEpochs(5)
#t.trainUntilConvergence(dataset=ds, maxEpochs=10000, verbose=True)
trnresult = percentError( t.testOnClassData(),ds['class'] )
testresult = percentError( t.testOnClassData(),ts['class'] )
print "epoch: %4d" % t.totalepochs,"  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % testresult

#guesses = []
#
#def one_or_zero(array):
#    return array[1]>.5
#
#for line in test_final:
#    guesses.append(one_or_zero(n.activate(line)))
#    final_guess = pd.DataFrame(guesses,columns=['requester_received_pizza'],dtype=int).join(test_ids)
#    final_guess.set_index('request_id', inplace=True)
#    final_guess.to_csv('/desktop/submission.csv')
