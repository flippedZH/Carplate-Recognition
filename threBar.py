import cv2 as cv

def threshold_trackbar(gray_res):
    cv.namedWindow('image', flags=cv.WINDOW_NORMAL)
    cv.createTrackbar('num', 'image', 0, 255, lambda x: None)
    while True:
        num = cv.getTrackbarPos('num', 'image')
        _, threshold1 = cv.threshold(gray_res, num, 255, cv.THRESH_BINARY)
        img = cv.resize(threshold1, (500, 500))
        cv.imshow('image',img)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

img = cv.imread("src5.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
threshold_trackbar(gray)