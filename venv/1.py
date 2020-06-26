import os


def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag


# 扫面文件
def GetFileList(FindPath, FlagStr=[]):
    FileList = []
    FileNames = os.listdir(FindPath)
    if len(FileNames) > 0:
        for fn in FileNames:
            if len(FlagStr) > 0:
                if IsSubString(FlagStr, fn):
                    fullfilename = os.path.join(FindPath, fn)
                    FileList.append(fullfilename)
            else:
                fullfilename = os.path.join(FindPath, fn)
                FileList.append(fullfilename)

    if len(FileList) > 0:
        FileList.sort()

    return FileList


train_txt = open('C:/Users/NEFU-HJP/Desktop/t.txt', 'w')
# 制作标签数据，多标签数据，编号从0开始
imgfile = GetFileList('D:/srt/test/images/w_0e737d0')  # .py文件目录下
for img in imgfile:
    str1 = img + ' ' + '0' + '\n'  # 用空格代替转义字符 \t
    train_txt.writelines(str1)

imgfile = GetFileList('D:/srt/test/images/w_1eafe46')
for img in imgfile:
    str2 = img + ' ' + '1' + '\n'
    train_txt.writelines(str2)

imgfile = GetFileList('D:/srt/test/images/w_2d99a0c')
for img in imgfile:
    str3 = img + ' ' + '2' + '\n'
    train_txt.writelines(str3)

imgfile = GetFileList('D:/srt/test/images/w_4b7b80b')
for img in imgfile:
    str4 = img + ' ' + '3' + '\n'
    train_txt.writelines(str4)

imgfile = GetFileList('D:/srt/test/images/w_4e52a49')
for img in imgfile:
    str5 = img + ' ' + '4' + '\n'
    train_txt.writelines(str5)

imgfile = GetFileList('D:/srt/test/images/w_6c803bf')
for img in imgfile:
    str6 = img + ' ' + '5' + '\n'
    train_txt.writelines(str6)

imgfile = GetFileList('D:/srt/test/images/w_7b035cc')
for img in imgfile:
    str7 = img + ' ' + '6' + '\n'
    train_txt.writelines(str7)

imgfile = GetFileList('D:/srt/test/images/w_7c7a78c')
for img in imgfile:
    str8 = img + ' ' + '7' + '\n'
    train_txt.writelines(str8)

imgfile = GetFileList('D:/srt/test/images/w_7e8b270')
for img in imgfile:
    str9 = img + ' ' + '8' + '\n'
    train_txt.writelines(str9)

# 转换完成后，将.txt文档关闭
train_txt.close()
