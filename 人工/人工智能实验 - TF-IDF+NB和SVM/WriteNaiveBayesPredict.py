import os
import time
import pickle
import numpy as np


from sklearn.utils import Bunch
from sklearn import metrics
from Tools import readBunchObj
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB # 导入多项式贝叶斯算法
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix


def metrics_result(actual, predict):
    print('正确:{0:.3f}'.format(metrics.accuracy_score(actual, predict)))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))

if __name__ == "__main__":
    time1 = time.process_time()
    print("开始训练...")
    # 导入训练集
    bunch = readBunchObj("./BunchOfTrainingSets/BunchOfTrainingSets.dat")
    tfidfSpace = Bunch(label = bunch.label, filenames = bunch.filenames, tdm = [], vocabulary = {})
    
    trainSpace = Bunch(label = bunch.label, filenames = bunch.filenames, tdm = [], vocabulary = {})

    vector = CountVectorizer()
    transformer = TfidfTransformer()
    train_tfidf = transformer.fit_transform(vector.fit_transform(bunch.contents))
    trainSpace.vocabulary = vector.vocabulary_
    word = vector.get_feature_names()
    weight = train_tfidf.toarray()
    
    # print(train_tfidf.nonzero())
    Index = train_tfidf.nonzero()
    tmpx = Index[0]
    tmpy = Index[1]

    # print(len(tmpx))

    n = len(weight)
    m = len(word)

    labels = dict()
    relabels = []
    Belong = np.zeros(len(weight))
    cnt = 0
    i = 0
    preCount = []
    for mylabel in tfidfSpace.label:
        if (mylabel not in labels):
            labels[mylabel] = cnt
            relabels.append(mylabel)
            preCount.append(0)
            cnt = cnt + 1
        Belong[i] = labels[mylabel]
        preCount[int(Belong[i])] = preCount[int(Belong[i])] + 1
        i = i + 1
    # print(cnt)
    # print(m)
    mat = np.zeros((cnt, m))
    sumval = np.zeros(cnt)

    for i in range(cnt):
        preCount[i] = preCount[i] / n
        # print(preCount[i])

    for id in range(len(tmpx)):
        i = tmpx[id]
        j = tmpy[id]
        mat[int(Belong[i])][j] = mat[int(Belong[i])][j] + weight[i][j]
        sumval[int(Belong[i])] = sumval[int(Belong[i])] + weight[i][j]

    absSum = 0
    for i in range(cnt):
        absSum = absSum + sumval[i]

    for i in range(cnt):
        for j in range(m):
            mat[i][j] = (mat[i][j] + 0.025)/ (sumval[i] + absSum)
            # print(mat[i][j], end= " ")
        # print("")
    # print(mat)
    
    reword = dict()
    for i in range(m):
        reword[word[i]] = i
    
    

    mat = np.log(mat)
    preCount = np.log(preCount)

    # for i in range(cnt):
    #         for j in range(m):
    #             print(mat[i][j])
    
    print("训练成功!")
    time2 = time.process_time()    
    print('运行时间: %s s\n\n' % (time2 - time1))
    
    # 导入测试集
    print("开始测试...")
    bunch = readBunchObj("./BunchOfTestSets/BunchOfTestSets.dat")
    tfidfSpace = Bunch(label = bunch.label, filenames = bunch.filenames, tdm = [], vocabulary = {})
    
    vector = CountVectorizer(vocabulary = trainSpace.vocabulary)
    transformer = TfidfTransformer()
    test_tfidf = transformer.fit_transform(vector.fit_transform(bunch.contents))
    word = vector.get_feature_names()
    weight = test_tfidf.toarray()
    
    Index = test_tfidf.nonzero()
    tmpx = Index[0]
    tmpy = Index[1]

    ActBelong = np.zeros(len(tfidfSpace.label))
    cnt1 = 0
    i1 = 0
    labels1 = dict()
    preCount1 = np.zeros(cnt)
    for mylabel in tfidfSpace.label:
        if (mylabel not in labels1):
            labels1[mylabel] = cnt1
            cnt1 = cnt1 + 1
        ActBelong[i1] = labels[mylabel]
        preCount1[int(ActBelong[i1])] += 1
        i1 = i1 + 1

    ntest = len(weight)
    n = len(weight)
    m = len(word)

    errNum = 0
    errNumList = np.zeros(cnt)
    
    corWords = []
    for i in range(ntest):
        corWords.append([])

    # for i in range(ntest):
    for j in range(len(tmpx)):    
        corWords[tmpx[j]].append(tmpy[j])

    
    confusion_matrix = np.zeros((cnt, cnt))

    actual = []
    predict = []
    for i in range(ntest): # 文章
        Max = -1000000000000
        Type = -1
        for k in range(cnt): # 所有类
            now = preCount[k]
            for j in corWords[i]:
                now = now + mat[k][j] * weight[i][j]
            if Max < now:
                Max = now
                Type = k
            # print(Max, Type)

        if Type != int(ActBelong[i]):
            errNumList[int(ActBelong[i])] += 1
            errNum += 1
            confusion_matrix[int(ActBelong[i])][Type] += 1
        else:
            confusion_matrix[int(ActBelong[i])][int(ActBelong[i])] += 1
        actual.append(ActBelong[i])
        predict.append(Type)

    print("混淆矩阵:")
    print(confusion_matrix)

    for j in range(cnt):
        tot = 0
        for i in range(cnt):
            tot += confusion_matrix[i][j]
        
        print("第", j, end = " ")
        print("类的召回率率:", (confusion_matrix[j][j])/tot)

    for i in range(cnt):
        print("第", i, end = " ")
        print("类的正确率:", (preCount1[i] - errNumList[i])/preCount1[i])
    
    metrics_result(actual, predict)
    print("测试结束\n\n")
    
                

    # for i in range(len(weight)):
    #     tot = 0
    #     print("----这里输出第"+str(i)+"类文本的词语tf-idf权重------")
    #     for j in range(len(word)):
    #         if weight[i][j] != 0.0:
    #             print(word[j],weight[i][j])
    #             tot = tot + weight[i][j]
        # print("tot=", tot)
        # print("\n\n")
    

    # print(train_tfidf_sets.label)
    # print(train_tfidf_sets.filenames)
    
    # print(train_tfidf_sets.vocabulary) # 永康 13814
    # word = vectorizer.get_feature_names()
    # print(tfidfSpace.tdm.toarray())
    # weight = tfidfSpace.tdm.toarray()
    # for i in range(len(weight)):
    #     print("----这里输出第"+str(i)+"类文本的词语tf-idf权重------")
    #     for j in range(len(word)):
    #         print(word[j],weight[i][j])

    '''

    # 导入测试集
    test_tfidf_path = "./BunchOfTestSets/test_tfidf_space.dat"
    test_tfidf_sets = readBunchObj(test_tfidf_path)

    # 训练分类器，输入词袋向量和分类标签
    # 朴素贝叶斯，NaiveBayes
    classifier = MultinomialNB(alpha = 0.000001).fit(train_tfidf_sets.tdm, train_tfidf_sets.label)
    
    # Support Vector Machine
    # classifier = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
    # classifier.fit(train_tfidf_sets.tdm, train_tfidf_sets.label)

    # 预测分析结果
    predictedResult = classifier.predict(test_tfidf_sets.tdm)

    # 错误信息
    print("错误信息如下:")
    for act_label, file_name, expct_label in zip(test_tfidf_sets.label, test_tfidf_sets.filenames, predictedResult):
        if act_label != expct_label:
            print(file_name, ": 实际类别:", act_label, "   预测类别:", expct_label)

    print("\n\n每类和总体正确率、召回率、f1-score如下")
    print(classification_report(test_tfidf_sets.label, predictedResult))
    '''
    time2 = time.process_time()    
    print('运行时间: %s s' % (time2 - time1))

