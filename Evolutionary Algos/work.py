# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from scipy import signal
from scipy import misc
from scipy import stats
import scipy
# load image as pixel array

mainFrame = Image.open('groupGray.jpg')
mainFrame = mainFrame.convert("L")

baba = Image.open("boothiGray.jpg")
baba = baba.convert("L")
print(baba.format)
print(baba.mode)
mrows,mcols = mainFrame.size
brows,bcols = baba.size
print(f"Baba Jee Rows : {bcols} baba jee cols : {brows}")
babaarray = asarray(baba)
mainarray = asarray(mainFrame)
print(stats.kendalltau(babaarray,mainarray[108:108+35,643:643+29]).correlation)

# mainarray = mainarray - mainarray.mean()
# babaarray = babaarray - babaarray.mean()
# corr = signal.correlate2d(mainarray, babaarray, boundary='symm', mode='same')
# plt.imshow(corr,cmap="gray")
# plt.show()
# print(corr)
# print(np.amax(corr))
# result = np.where(corr == np.amax(corr))
# print(list(result))
# print("asdasda")
cropMain = mainarray[107:107+35,634:634+29]
print(mainarray.shape)
print(mainarray.ndim)
print(mainFrame.size)

print(baba.size)
print(babaarray.shape)
# in np mainarray max dimension (512,1024)
# 512 no of rows and 1024 nmbr of cols


# print("Core")
# r = np.corrcoef(cropMain, babaarray)

# print(type(np.mean(r,axis=(0,1))))
# print("Coreeee")
plt.imshow(mainFrame,cmap="gray")

plt.scatter(634, 107, s=100, c='red', marker='o')

plt.show()

plt.imshow(baba,cmap="gray")
plt.show()

def find_array(template,piece,solution):
    x,y = piece.shape
    startx,starty= solution
    print(startx,starty)
    # if startx+x < (template.shape)[0] and starty+y < (template.shape)[1]:
    result = template[startx:startx+x,starty:starty+y]
    main = []

    return
    for row in range(x):
        rows = []
        for col in range(y):
            # g = float("{:.1f}".format(result[row,col]/template[row,col])) 
            if result[row,col] == 0 or piece[row,col] == 0:
                g = 0
            elif  result[row,col] >= piece[row,col]:
                g = (piece[row,col] * 100)// result[row,col]
                g = float("{:.1f}".format(g/100))
            else:
                g = (result[row,col] * 100)// piece[row,col]
                g = float("{:.1f}".format(g/100))
            rows.append(g)
        main.append(rows)
    main = np.mean(main,axis=None)
    return main
         

# res = find_array(mainarray,babaarray,[122,653])
# # print(res)

x = mainarray[118:118+35,651:651+29]
y = np.copy(babaarray)
tau, p_value = stats.kendalltau(x, y)
