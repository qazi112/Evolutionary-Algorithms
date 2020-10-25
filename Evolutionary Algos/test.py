# import numpy as np
# from astroML.correlation import two_point
# np.random.seed(0)
# X = np.random.random((5000, 2))
# print(X)
# bins = np.linspace(0, 1, 20)
# print(bins)
# corr = two_point(X, bins)
# print(np.allclose(corr, 0, atol=0.02))


# Program for finding corner points of a plank

import cv2
import numpy as np

im = cv2.imread("plank.jpg")
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
gray = cv2.GaussianBlur(gray, (5, 5), 0)
print(gray)
_, bin = cv2.threshold(gray,120,255,1) # inverted threshold (light obj on dark bg)
bin = cv2.dilate(bin, None)  # fill some holes
bin = cv2.dilate(bin, None)
bin = cv2.erode(bin, None)   # dilate made our shape larger, revert that
bin = cv2.erode(bin, None)
contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Countors")

print(contours)
rc = cv2.minAreaRect(contours[0])
print("RC HAI")
print(rc)

box = cv2.boxPoints(rc)
print("BOX")
print(box)
for p in box:
    pt = (p[0],p[1])
    print(pt)
    cv2.circle(im,pt,5,(200,0,0),2)
cv2.imshow("plank.jpg", im)
cv2.waitKey()