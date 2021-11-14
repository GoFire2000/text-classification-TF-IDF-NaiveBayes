import pickle

# 保存至文件
def saveFile(savePath, content):
    with open(savePath, "wb") as fp:
        fp.write(content)

# 读取文件
def readFile(Path):
    with open(Path, "rb") as fp:
        content = fp.read()
    return content

# 读取bunch类型
def readBunchObj(path):
    with open(path, "rb") as fileobj:
        bunch = pickle.load(fileobj)
    return bunch

# 把bunch类型写入文件
def writeBunchObj(path, bunchobj):
    with open(path, "wb") as fileobj:
        pickle.dump(bunchobj, fileobj)
