from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
import  cv2 as cv
import cv2
import numpy as np
from sklearn.externals import joblib

def  GetRes_plate(box):
    gray = cv2.cvtColor(box,cv2.COLOR_BGR2GRAY)
    character_Arr = np.zeros((1,3600))
    img = cv2.resize(gray, (120, 30), interpolation=cv2.INTER_LINEAR)
    new_character_ = img.reshape((1,3600))[0]
    character_Arr[0,:] =  new_character_
    clf = joblib.load("based_SVM_plate_train_model.m")
    predict_result = clf.predict(character_Arr)
    middle_route = ['0', '1']
    return middle_route[predict_result.tolist()[0]]