from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
import cv2 as cv
import cv2
import numpy as np
import os
from sklearn.externals import joblib


def  GetRes_char(box_char):
    gray = cv2.cvtColor(box_char,cv2.COLOR_BGR2GRAY)
    #gray=box_char
    character_Arr = np.zeros((1,400))
    #print(len(character_list))
    img = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_LINEAR)  ##还原为原来的大小  INTER_LINEAR 线性插值
    new_character_ = img.reshape((1,400))[0]
    character_Arr[0,:] =  new_character_

    clf = joblib.load("based_SVM_character_train_model.m")
    predict_result = clf.predict(character_Arr)
    middle_route = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', \
                    'G', 'H', 'J', 'K','L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    #print(predict_result.tolist())  #将数组或者矩阵转换成列表
    # for k in range(len(predict_result.tolist())):
    #     print('%c'%middle_route[predict_result.tolist()[k]])  #结果

    return middle_route[predict_result.tolist()[0]]

def  GetRes_char_province(box_char):
    gray = cv2.cvtColor(box_char,cv2.COLOR_BGR2GRAY)
    character_Arr = np.zeros((1,400))
    img = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_LINEAR)  ##还原为原来的大小  INTER_LINEAR 线性插值
    new_character_ = img.reshape((1,400))[0]
    character_Arr[0,:] =  new_character_
    clf = joblib.load("based_SVM_province_train_model.m")
    predict_result = clf.predict(character_Arr)
    ProvinceList=['川','鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '山西', '苏', '晋',
                  '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
    return ProvinceList[predict_result.tolist()[0]]

def JoinChar():
    img_path="CutChar_Img"
    filename_list = os.listdir(img_path)
    res=[]
    for i in range(len(filename_list)):
        if(i==0):
            TempPath="CutChar_Img/"+str(i+1)+".jpg"
            box_char=cv.imread(TempPath)
            res.append(GetRes_char_province( box_char))
        elif(i==2):
            pass
        else:
            TempPath="CutChar_Img/"+str(i+1)+".jpg"
            box_char=cv.imread(TempPath)
            res.append(GetRes_char( box_char))
    return res