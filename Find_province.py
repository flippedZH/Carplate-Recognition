
import numpy as np
import cv2
import  cv2 as cv
import  os


def load_data(filename_1):
    filepath2 = "C:\\Users\\zh\\Desktop\\data_province\\txt\\"
    pathlist=os.listdir(filepath2)
    path_name=[]
    for i in pathlist:
        path_name.append(i.split(".")[0])
    with open(filename_1, 'r') as fr_1:
        temp_address = [row.strip() for row in fr_1.readlines()]
    middle_route=path_name

    sample_number = 0  # 用来计算总的样本数
    dataArr = np.zeros((3216, 400))
    label_list = []

    for i in range(len(temp_address)):

        with open(r'C:\Users\zh\Desktop\data_province\txt\\' + temp_address[i]+'.txt', 'r') as fr_2:

            temp_address_2 = [row_1.strip() for row_1 in fr_2.readlines()]  ##？？？？？？？？

        for j in range(len(temp_address_2)):
            sample_number += 1

            path_temp="C:/Users\zh\Desktop\data_province\data/" +str(middle_route[i]) + '/' + str(temp_address_2[j])
            temp_img = cv2.imread(path_temp)
            temp_img=cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)

            print(sample_number)
            #temp_img=cv.cvtColor(temp_img,cv2.COLOR_BAYER_GR2GRAY)
            temp_img = cv2.resize(temp_img, (20, 20), interpolation=cv2.INTER_LINEAR)

            temp_img = temp_img.reshape(1, 400)
            #前面定义了的  表示 第i行的所有列用读取的图片temp_img填充
            dataArr[sample_number - 1, :] = temp_img   # dataArr = np.zeros((13156, 400))
        #  label_list = []
        #给每个标签扩充 数量（因为同个文件夹里面的标签相同）
        label_list.extend([i] * len(temp_address_2))

    return dataArr, np.array(label_list)  #返回（13156, 400）数据集中的所有数据以及对应的标签 标签格式是一样的

def SVM_rocognition_character(img): ## character_list是输入的图片 可能是多张图片
    character_Arr = np.zeros((1,400))
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LINEAR)  ##还原为原来的大小  INTER_LINEAR 线性插值
    new_character_ = img.reshape((1,400))[0]
    character_Arr[0,:] =  new_character_

    from sklearn.externals import joblib
    clf = joblib.load("based_SVM_province_train_model.m")
    predict_result = clf.predict(character_Arr)

    #print(predict_result.tolist())  #将数组或者矩阵转换成列表 predict_result中存的是序号
    ProvinceList=['川','鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '山西', '苏', '晋',
                  '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
    #for k in range(len(predict_result.tolist())):
    #     print('%c'%ProvinceList[predict_result.tolist()[k]])  #结果
    return predict_result.tolist()[0]

if __name__=="__main__":
    src=cv2.imread("CutChar_Img/1.jpg")
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    SVM_rocognition_character(gray)