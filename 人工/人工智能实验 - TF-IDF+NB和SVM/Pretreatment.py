# 对训练集和测试集进行分词,并去除停等词

import os
import time
import jieba
import multiprocessing
import jieba.posseg as psg
from Tools import saveFile, readFile

def FenciProcess(initial_path, target_path):
    # 停等词,stop_word_list里面存的是所有的停等词
    # stop_words_list = []
    stop_words_dict = dict()
    with open('./stop_words.txt', 'r', encoding = 'utf-8') as fp:
        while True:
            word = fp.readline()
            if word == '':
                break
            stop_words_dict[word[:-1]] = 1
            # stop_words_list.append(word[ : -1])

    # stop_words_list.sort()
    # for words in stop_words_list:
    #     print(words)

    print("开始分词...")
    # initialPath是待处理的语料库路径，targetPath是处理后语料库存储路径
    dir_list = os.listdir(initial_path)
    
    for mydir in dir_list:
        ini_path = initial_path + mydir + "/utf8/"
        tar_path = target_path + mydir + "/"

        if not os.path.exists(tar_path):
            os.makedirs(tar_path)
        
        file_list = os.listdir(ini_path)

        for myfile in file_list:
            fullname = ini_path + myfile
            content = readFile(fullname)
            # 删除换行、空行和多余空格
            content = content.replace('\r\n'.encode('utf-8'), ''.encode('utf-8')).strip()
            content = content.replace(' '.encode('utf-8'), ''.encode('utf-8')).strip()

            # 调用jieba，进行分词
            content_fenci = jieba.posseg.cut(content)
            fenci_list = []
            for x in content_fenci:
                if 'n' in x.flag:
                    fenci_list.append(x.word)
            newContent = []
            for myword in fenci_list:
                if (myword not in stop_words_dict):
                    newContent.append(myword)

            # 保存到目标路径
            saveFile(tar_path + myfile, ' '.join(newContent).encode('utf-8'))
    print('分词结束!')

if __name__ == "__main__":

    time1 = time.process_time()

    # 对训练集进行分词处理
    print("对训练集进行分词和去停处理...")
    initial_sets_path = "./TrainingSets/" # 未分词的训练集路径
    target_sets_path = "./ProcessedTrainingSets/" # 分词后训练集路径
    FenciProcess(initial_sets_path, target_sets_path)

    time2 = time.process_time()
    print('运行时间: %s s\n\n' % (time2 - time1))

    # 对测试集进行分词处理
    print("对测试集进行分词和去停处理...")
    initial_sets_path = "./TestSets/"
    target_sets_path = "./ProcessedTestSets/"
    FenciProcess(initial_sets_path, target_sets_path)

    time3 = time.process_time()
    print('运行时间: %s s' % (time3 - time2))
