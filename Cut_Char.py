import cv2
import cv2 as cv
import numpy as np

def CutChar(BinImg):
    # cv.imshow("BinImg",BinImg)
    #     # cv.waitKey(0)
    #gray = cv2.cvtColor(BinImg,cv2.COLOR_BGR2GRAY)
    gray=BinImg
    gray=cv.resize(gray,(366,72))
    CutMark=[]
    for i in range(gray.shape[1]): ## 长
        num=0
        for j in range(gray.shape[0]): # 高
            if(gray[j][i]>=0) and (gray[j][i]<20):
                num=num+1
            if(num>70):
                if i not in CutMark:
                    CutMark.append(i)

    CutMark_push1=[]
    CutMark_push2=[]

    lengthMark=len(CutMark)
    CutMark.append(0)

    for i in range(lengthMark):
        if((CutMark[i]+1)==CutMark[i+1]):
            CutMark_push1.append(CutMark[i])
            if((CutMark_push1[0]-0<5)):
                CutMark_push1.clear()
            elif ((gray.shape[1]-CutMark_push1[len(CutMark_push1)-1])<5):
                CutMark_push1.clear()
        elif(len(CutMark_push1)!=0):
            CutMark_push2.append(CutMark_push1[int(len(CutMark_push1)/2)+1])
            CutMark_push1.clear()


    #print("res",CutMark_push2)

    lengthCutMark=len(CutMark_push2)

    CutMark_push2.append(gray.shape[1])
    CutMark_push2.reverse()
    CutMark_push2.append(0)
    CutMark_push2.reverse()
   # print(CutMark_push2)

    for i in range(lengthCutMark+1):
        CutImg=gray[:,CutMark_push2[i]:CutMark_push2[i+1]]
        cv.imwrite("CutChar_Img/"+str(i+1)+".jpg",CutImg)

if __name__ == '__main__':

    #cv.imshow("src",gray)
    print(gray.shape)
    #cv.waitKey(0)

