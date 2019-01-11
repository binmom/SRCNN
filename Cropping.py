import os
import cv2
import numpy as np
import random

def listFiles(path, extension, extension2):
    return [f for f in os.listdir(path) if f.endswith(extension) or f.endswith(extension2)]
image = listFiles('Train','.jpg', '.bmp')
image2 = listFiles('Test','.jpg','.bmp')

# print(image2)

name = 16402
for i in image:
	img = cv2.imread('./Train/'+i)
	seed_h, seed_r, n = img.shape
	# print(min(seed_h,seed_r))
	r = 28
	h = 28
	while h < seed_h-40:
		while r < seed_r-40:
			new_img = img[h:h+32, r:r+32]
		    # new_img = cv2.GaussianBlur(img,(t,t),cv2.BORDER_DEFAULT)
		    # print(new_img)
			cv2.imwrite(os.path.join('./Train_Cropped/',str(name) +'.jpg'),new_img)
			name += 1
			r += 32
			print(h,r)
		h += 32
		r = 0

