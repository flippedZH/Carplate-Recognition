# -*- coding: utf-8 -*-
"""
Created on Saturday June 27  12:16:30 2020
@author: ZouHeng
"""

import cv2
import cv2 as cv
import numpy as np
import Pers
import CutRoom
import Find_plate
import Cut_Char
import Find_char

def GetBinImg(SrcPath):
    src=cv.imread(SrcPath)
    src=cv.resize(src,(0,0),fx=0.5,fy=0.5)
    #灰度化处理
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    ##滤波处理  高斯
    blur = cv2.GaussianBlur(gray,(11,11),0)

    #形态学处理
    #OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15)) ##15最合适

    eroded = cv2.erode(blur,kernel) #去掉图片中的细节

    # 扩大主要 像素
    dilate = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    #阈值分割
    #(t,thresh1) = cv2.threshold(gray_img,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_TRIANGLE)
    otsuThe, dst_Otsu = cv2.threshold(dilate,72, 255, cv2.THRESH_OTSU)

    #边缘检查
    img_canny= cv2.Canny(dst_Otsu, 3,0, 150)  # 用Canny函进行边缘检测

    # 图像 阀值下限 阀值上限 算子内核大小
    thresholdImage, contours, hierarchy = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ##轮廓点接收
    cnt=[]
    for i in range(len(contours)):
        #print("第"+str(i)+"个轮廓",contours[i])
        cnt.append(np.array(contours[i]).reshape(len(contours[i]),2))

    # 画出带有倾角的矩形
    for i in range(len(cnt)):
        rect = cv2.minAreaRect(cnt[i])
        # minAreaRect：rect是一个由三个元素组成的元组，依次代表旋转矩形的中心点坐标、尺寸和旋转角度（根据中心坐标、尺寸和旋转角度
        # 可以确定一个旋转矩形）
        box = cv.boxPoints(rect)
        #boxPoints：根据旋转矩形的中心的坐标、尺寸和旋转角度，计算出旋转矩形的四个顶点
        box = np.int0(box)

        w=(((box[1][0]-box[0][0])**2)+((box[1][1]-box[0][1])**2))**0.5
        l=(((box[3][0]-box[0][0])**2)+((box[3][1]-box[0][1])**2))**0.5

        if (l!=0) and  (w!=0):
            Ration=l/w
        if((Ration<5.5 and Ration>2.5) or (1/Ration<5.5 and 1/Ration>2.5)):
            cv2.drawContours(src, [box], 0, (0, 0, 255), 2)
            example=Pers.PerspectiveTrans(box,src)
            CutImg=example.FourPointsTransform()
            Res_plate=Find_plate.GetRes_plate(CutImg)

            if(Res_plate=="1"):
                cv.imwrite("img_plate"+".png", CutImg)
                return  CutImg

if __name__ == "__main__":
    SrcPath="src.jpg"
    CarPlate1=GetBinImg(SrcPath)
    if (CarPlate1.any==None):
        print("未检测到车牌")
    else:
        BinImg=CutRoom.remove_plate_upanddown_border(CarPlate1)
        Cut_Char.CutChar(BinImg)
        res=Find_char.JoinChar()
        print(res)

