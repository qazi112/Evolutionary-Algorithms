# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray



# load image as pixel array

mainFrame = Image.open('groupGray.jpg')
baba = Image.open("boothiGray.jpg")
print(baba.format)
print(baba.mode)
mrows,mcols = mainFrame.size
brows,bcols = baba.size
print(f"Baba Jee Rows : {bcols} baba jee cols : {brows}")
babaarray = asarray(baba)
mainarray = asarray(mainFrame)








# print(type(mainarray[:35,:29]))
# print(type(babaarray[:35,:29]))

cropMain = mainarray[:35,:29]
print(mainarray.shape)
print(mainarray.ndim)
print(mainFrame.size)
# in np mainarray max dimension (512,1024)
# 512 no of rows and 1024 nmbr of cols
print(mainarray[107,634])
r = np.corrcoef(cropMain, babaarray)
# print(r)
# print(r[0])
# print(np.mean(r,axis=(0,1)))

plt.imshow(mainFrame,cmap="gray")

plt.scatter(634, 107, s=100, c='red', marker='o')

plt.show()

plt.imshow(baba)
plt.show()

