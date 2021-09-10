import cv2 as cv
import numpy as np

class PerspectiveTrans(object):
    def __init__(self, pts, img):
        self.pts = pts
        self.img = img
    # 中心点排序编号函数：四个点按照左上、右上、右下、左下的顺序排列（顺时针）
    def OrderPoints(self):
        rect = np.zeros((4, 2), dtype="float32") # 创建4行2列的全零二维数组
        s = self.pts.sum(axis=1)                 # 数组元素横着相加：将每个点的横纵坐标相加
        rect[0] = self.pts[np.argmin(s)]         # 上一步值最小的编号为第一个点
        rect[2] = self.pts[np.argmax(s)]         # 上一步值最大的编号为第四个点
        diff = np.diff(self.pts, axis=1)         # 数组元素横着做差：将每个点的横纵坐标相减（右减左）
        rect[1] = self.pts[np.argmin(diff)]      # 上一步值最小的编号为第二个点
        rect[3] = self.pts[np.argmax(diff)]      # 上一步值最大的编号为第三个点
        return rect                              # 返回排好序的中心点

    def FourPointsTransform(self):  # image 为需要透视变换的图像  pts为四个点
        rect = self.OrderPoints()  # 四点排序
        (tl, tr, br, bl) = rect   # 用元组分别接收数组元素
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)) # 计算宽度，用于透视变换
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))    # 保留最大的宽度
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))# 计算高度，用于透视变换
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB)) # 保留最大的宽度
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32") # 构建变换之后的目标点
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(self.img, M, (maxWidth, maxHeight)) #以四点的变换关系为基准，变换所有的像素点
        return warped
