import os
import time
import pickle

from sklearn.utils import Bunch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readFile, readBunchObj, writeBunchObj
from sklearn.decomposition import LatentDirichletAllocation


def vector_space(bunch_path, space_path, train_lda_path = None):
    print("开始LDA构建词模型...")
    bunch = readBunchObj(bunch_path)
    ldaSpace = Bunch(label = bunch.label, filenames = bunch.filenames, tdm = [], vocabulary = {})

    if train_lda_path is None:
        # 对训练集进行处理
        vectorizer = CountVectorizer()
        counts_train = vectorizer.fit_transform(bunch.contents)
        ldaSpace.vocabulary = vectorizer.vocabulary_
        lda = LatentDirichletAllocation(n_components = 10, max_iter = 50, learning_method = 'batch')
        ldaSpace.tdm = lda.fit(counts_train).transform(counts_train)
    else:
        # 对测试集进行处理，利用训练集的信息
        trainBunch = readBunchObj(train_lda_path)
        ldaSpace.vocabulary = trainBunch.vocabulary
        vectorizer = CountVectorizer(vocabulary = trainBunch.vocabulary)
        counts_test = vectorizer.fit_transform(bunch.contents)
        lda = LatentDirichletAllocation(n_components = 10, max_iter = 50, learning_method = 'batch')
        ldaSpace.tdm = lda.fit(counts_test).transform(counts_test)

    writeBunchObj(space_path, ldaSpace)
    print("LDA构建词模型结束!")

if __name__ == '__main__':
    time1 = time.process_time()
    # 对训练集进行处理,LDA构建词模型
    bunch_path = "./BunchOfTrainingSets/BunchOfTrainingSets.dat"
    space_path = "./BunchOfTrainingSets/train_lda_spcae2.dat"
    vector_space(bunch_path, space_path)

    time2 = time.process_time()
    print('运行时间: %s s\n\n' % (time2 - time1))

    # 对测试集进行处理，LDA构建词模型，利用训练集的信息
    bunch_path = "./BunchOfTestSets/BunchOfTestSets.dat"
    space_path = "./BunchOfTestSets/test_lda_space2.dat"
    train_lda_path = "./BunchOfTrainingSets/train_lda_spcae2.dat"
    vector_space(bunch_path, space_path, train_lda_path)
    
    time3 = time.process_time()
    print('运行时间: %s s' % (time3 - time2))