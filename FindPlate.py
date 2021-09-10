############################机器学习识别字符##########################################
#这部分是支持向量机的代码
import numpy as np
import cv2

##数据说明：
##标签0:8000个
##标签1:2800个
##图片格式：120*30

def load_data(filename_1):
    with open(filename_1, 'r') as fr_1:
        temp_address = [row.strip() for row in fr_1.readlines()]
    middle_route = ['0', '1']
    sample_number = 0  # 用来计算总的样本数
    dataArr = np.zeros((8000+2800, 3600))
    label_list = []
    for i in range(len(temp_address)):
        with open(r'C:\Users\zh\Desktop\data_plate\txt\\' + temp_address[i]+'.txt', 'r') as fr_2:

            temp_address_2 = [row_1.strip() for row_1 in fr_2.readlines()]  ##？？？？？？？？
        # print(temp_address_2)
        # sample_number += len(temp_address_2)
        for j in range(len(temp_address_2)):
            sample_number += 1

            path_temp="C:/Users\zh\Desktop\data_plate\data/" +str(middle_route[i]) + '/' + str(temp_address_2[j])
            temp_img = cv2.imread(path_temp, cv2.COLOR_BGR2GRAY)
            temp_img = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
            temp_img = cv2.resize(temp_img, (120, 30), interpolation=cv2.INTER_LINEAR)
            temp_img = temp_img.reshape(1, 3600)

            #前面定义了的  表示 第i行的所有列用读取的图片temp_img填充
            dataArr[sample_number - 1, :] = temp_img   # dataArr = np.zeros((13156, 400))

        label_list.extend([i] * len(temp_address_2))

    return dataArr, np.array(label_list)  #返回（13156, 400）数据集中的所有数据以及对应的标签 标签格式是一样的

def SVM_rocognition(dataArr, label_list):
    from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
    estimator = PCA(n_components=20)  # 初始化一个可以将高维度特征向量（400维）压缩至20个维度的PCA   （13156, 20）？
    new_dataArr = estimator.fit_transform(dataArr)   ##？？
    import sklearn.svm
    svc = sklearn.svm.SVC()
    svc.fit(dataArr, label_list)           # 没有降维，直接训练         #就是这一步                                                                                                                                                                                 # 使用默认配置初始化SVM，对原始400维像素特征的训练数据进行建模，并在测试集上做出预测
    from sklearn.externals import joblib  # 通过joblib的dump可以将模型保存到本地，clf是训练的分类器
    joblib.dump(svc,"based_SVM_plate_train_model.m")  # 保存训练好的模型，通过svc = joblib.load("based_SVM_character_train_model.m")调用

def SVM_rocognition_character(img): ## character_list是输入的图片 可能是多张图片
    character_Arr = np.zeros((1,3600))
    #print(len(character_list))
    img = cv2.resize(img, (120, 30), interpolation=cv2.INTER_LINEAR)  ##还原为原来的大小  INTER_LINEAR 线性插值
    new_character_ = img.reshape((1,3600))[0]
    character_Arr[0,:] =  new_character_

    from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
    estimator = PCA(n_components=20)  # 初始化一个可以将高维度特征向量（400维）压缩至20个维度的PCA

    filename_1 = r'C:\Users\zh\Desktop\data_plate\txt\dizhi.txt'
    dataArr, label_list = load_data(filename_1)
    SVM_rocognition(dataArr, label_list)

    from sklearn.externals import joblib
    clf = joblib.load("based_SVM_plate_train_model.m")
    predict_result = clf.predict(character_Arr)

    middle_route = ['0', '1']
    #print(predict_result.tolist())  #将数组或者矩阵转换成列表
    # for k in range(len(predict_result.tolist())):
    #     print('%c'%middle_route[predict_result.tolist()[k]])  #结果

if __name__=="__main__":
    src=cv2.imread("img_5.jpg")
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    SVM_rocognition_character(gray)