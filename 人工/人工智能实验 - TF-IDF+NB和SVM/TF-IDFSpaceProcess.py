import os
import time
import pickle

from sklearn.utils import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readFile, readBunchObj, writeBunchObj

def vector_space(bunch_path, space_path, train_tfidf_path = None):
    print("开始生成TF-IDF词向量空间...")
    bunch = readBunchObj(bunch_path)
    tfidfSpace = Bunch(label = bunch.label, filenames = bunch.filenames, tdm = [], vocabulary = {})

    if train_tfidf_path is None:
        # 对训练集进行处理，生成训练集的tf-idf词向量空间
        vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5)
        tfidfSpace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfSpace.vocabulary = vectorizer.vocabulary_
    else:
        # 对测试集进行处理，生成测试集的tf-idf词向量空间，利用训练集的信息
        trainBunch = readBunchObj(train_tfidf_path)
        tfidfSpace.vocabulary = trainBunch.vocabulary
        vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, vocabulary = trainBunch.vocabulary)
        tfidfSpace.tdm = vectorizer.fit_transform(bunch.contents)
        
    writeBunchObj(space_path, tfidfSpace)
    print("TF-IDF词向量空间生成完毕!")

if __name__ == '__main__':
    time1 = time.process_time()
    # 对训练集进行处理，生成训练集的tf-idf词向量空间
    bunch_path = "./BunchOfTrainingSets/BunchOfTrainingSets.dat"
    space_path = "./BunchOfTrainingSets/train_tfidf_spcae.dat"
    vector_space(bunch_path, space_path)

    time2 = time.process_time()
    print('运行时间: %s s\n\n' % (time2 - time1))

    # 对测试集进行处理，生成测试集的tf-idf词向量空间，利用训练集的信息
    bunch_path = "./BunchOfTestSets/BunchOfTestSets.dat"
    space_path = "./BunchOfTestSets/test_tfidf_space.dat"
    train_tfidf_path = "./BunchOfTrainingSets/train_tfidf_spcae.dat"
    vector_space(bunch_path, space_path, train_tfidf_path)
    
    time3 = time.process_time()
    print('运行时间: %s s' % (time3 - time2))