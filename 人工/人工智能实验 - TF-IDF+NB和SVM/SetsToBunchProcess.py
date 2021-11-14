import os
import time
import pickle

from sklearn.utils import Bunch
from Tools import readFile, writeBunchObj

def SetsToBunchProcess(initial_path, target_path):
    print("开始构建文本对象...")
    # 获取分词处理后的集合的所有子目录
    dir_list = os.listdir(initial_path)

    # 创建一个存储信息的Bunch类
    # 分类（子目录），文件全路径，文件内容
    bunch = Bunch(label = [], filenames = [], contents = [])

    for mydir in dir_list:
        ini_path = initial_path + mydir + "/" #子目录的子目录
        file_list = os.listdir(ini_path) # ini_path下所有文件，也就是训练集和测试集
        
        for myfile in file_list:
            fullname = ini_path + myfile
            bunch.label.append(mydir) # 类别
            bunch.filenames.append(fullname) # 全路径
            bunch.contents.append(readFile(fullname)) # 文件内容
    
    writeBunchObj(target_path, bunch)
    print("构建文本对象结束！")


if __name__ == "__main__":
    time1 = time.process_time()

    # 对训练集进行Bunch化操作，生成.dat数据文件
    print("对训练集Bunch化，构造文本对象...")
    training_sets_path = "./ProcessedTrainingSets/"
    target_bunch_path = "./BunchOfTrainingSets/BunchOfTrainingSets.dat"
    SetsToBunchProcess(training_sets_path, target_bunch_path)

    time2 = time.process_time()
    print('运行时间: %s s\n\n' % (time2 - time1))

    # 对测试集进行Bunch化操作，生成.dat数据文件
    print("对测试集Bunch化，构造文本对象...")
    training_sets_path = "./ProcessedTestSets/"
    target_bunch_path = "./BunchOfTestSets/BunchOfTestSets.dat"
    SetsToBunchProcess(training_sets_path, target_bunch_path)

    time3 = time.process_time()
    print('运行时间: %s s' % (time3 - time2))