############################机器学习识别字符##########################################
#这部分是支持向量机的代码
import numpy as np
import cv2
import  cv2 as cv
import  os

filepath2 = "C:\\Users\\zh\\Desktop\\data_province\\txt\\"
pathlist=os.listdir(filepath2)
path_name=[]
for i in pathlist:
    path_name.append(i.split(".")[0])


def load_data(filename_1):
    """
    这个函数用来加载数据集，其中filename_1是一个文件的绝对地址
    """
    #读文件
    with open(filename_1, 'r') as fr_1:
        #文件夹中图片的绝对地址 记录 不是图片  而是文件地址
        temp_address = [row.strip() for row in fr_1.readlines()]
        #readlines:#依次读取每行
        #strip():去掉每行头尾空白
        # print(temp_address)
        # print(len(temp_address))
    # middle_route = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    #                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
    middle_route=path_name

    sample_number = 0  # 用来计算总的样本数
    dataArr = np.zeros((3216, 400))
    label_list = []

    #这里是两层循环：就是说 是读取 所有文件夹里面的 所有图片文件名字
    for i in range(len(temp_address)):
        # 这个就是完整地址
        #感觉是在读取一个txt文件，然后获取txt中图片对应的完整地址
        with open(r'C:\Users\zh\Desktop\data_province\txt\\' + temp_address[i]+'.txt', 'r') as fr_2:
            #相同的操作 因为是二次进入txt(?)文件
            temp_address_2 = [row_1.strip() for row_1 in fr_2.readlines()]  ##？？？？？？？？
        # print(temp_address_2)
        # sample_number += len(temp_address_2)
        for j in range(len(temp_address_2)):
            sample_number += 1
            # print(middle_route[i])
            # print(temp_address_2[j])

            ##根据上述完整地址中的 图片名？ 读取 灰度片图
            path_temp="C:/Users\zh\Desktop\data_province\data/" +str(middle_route[i]) + '/' + str(temp_address_2[j])
            temp_img = cv2.imread(path_temp)
            temp_img=cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
            # print('C:\\Users\Administrator\Desktop\python code\OpenCV\plate recognition\train\chars2\chars2\\'+ middle_route[i]+ '\\' +temp_address_2[j] )
            # cv2.imshow("temp_img",temp_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #变形
            print(sample_number)
            #temp_img=cv.cvtColor(temp_img,cv2.COLOR_BAYER_GR2GRAY)
            temp_img = cv2.resize(temp_img, (20, 20), interpolation=cv2.INTER_LINEAR)

            temp_img = temp_img.reshape(1, 400)
            #前面定义了的  表示 第i行的所有列用读取的图片temp_img填充
            dataArr[sample_number - 1, :] = temp_img   # dataArr = np.zeros((13156, 400))
        #  label_list = []
        #给每个标签扩充 数量（因为同个文件夹里面的标签相同）
        label_list.extend([i] * len(temp_address_2))
    # print(label_list)
    # print(len(label_list))
    return dataArr, np.array(label_list)  #返回（13156, 400）数据集中的所有数据以及对应的标签 标签格式是一样的

def SVM_rocognition(dataArr, label_list):
    from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
    estimator = PCA(n_components=20)  # 初始化一个可以将高维度特征向量（400维）压缩至20个维度的PCA   （13156, 20）？
    new_dataArr = estimator.fit_transform(dataArr)   ##？？
    ## new_testArr = estimator.fit_transform(testArr)  ？？

    import sklearn.svm
    svc = sklearn.svm.SVC()
    svc.fit(dataArr, label_list)           # 没有降维，直接训练         #就是这一步                                                                                                                                                                                 # 使用默认配置初始化SVM，对原始400维像素特征的训练数据进行建模，并在测试集上做出预测
    from sklearn.externals import joblib  # 通过joblib的dump可以将模型保存到本地，clf是训练的分类器
    joblib.dump(svc,"based_SVM_province_train_model.m")  # 保存训练好的模型，通过svc = joblib.load("based_SVM_character_train_model.m")调用

def SVM_rocognition_character(img): ## character_list是输入的图片 可能是多张图片
    character_Arr = np.zeros((1,400))
    #print(len(character_list))
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LINEAR)  ##还原为原来的大小  INTER_LINEAR 线性插值
    new_character_ = img.reshape((1,400))[0]
    character_Arr[0,:] =  new_character_

    from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
    estimator = PCA(n_components=20)  # 初始化一个可以将高维度特征向量（400维）压缩至20个维度的PCA
    #character_Arr = estimator.fit_transform(character_Arr)
    ############
    filename_1 = r'C:\Users\zh\Desktop\data_province\dizhi.txt'
    dataArr, label_list = load_data(filename_1)
    SVM_rocognition(dataArr, label_list)
    ##############
    from sklearn.externals import joblib
    clf = joblib.load("based_SVM_province_train_model.m")
    predict_result = clf.predict(character_Arr)
    # print("character_Arr",character_Arr)
    # print("character_Arr",character_Arr)
    middle_route=path_name
    # middle_route = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', \
    #                 'G', 'H', 'J', 'K','L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    print(predict_result.tolist())  #将数组或者矩阵转换成列表 predict_result中存的是序号
    ProvinceList=['川','鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '山西', '苏', '晋',
                  '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
    #
    #
    for k in range(len(predict_result.tolist())):
        print('%c'%ProvinceList[predict_result.tolist()[k]])  #结果

if __name__=="__main__":
    src=cv2.imread("CutChar_Img/1.jpg")
    #src=cv2.imread("C:\\Users\zh\Desktop\\1.jpg")

    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    # cv.imshow("?",src)
    # cv.waitKey(0)
    SVM_rocognition_character(gray)