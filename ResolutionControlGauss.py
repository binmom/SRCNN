import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt

def listFiles(path, extension, extension2):
    return [f for f in os.listdir(path) if f.endswith(extension) or f.endswith(extension2)]
# image = listFiles('Test','.jpg', '.png')
# image2 = listFiles('Train_Cropped','.jpg','.png')
test1 = listFiles('Test','.jpg', '.bmp')

# print(image2)
for i in test1:
    img = cv2.imread('./Test/'+i)
    print(img)
    # print(img)
    new_img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    
    # print(new_img)
    cv2.imwrite(os.path.join('C:\\Users\\user\\Desktop\\study\\pytorch\\Project_1 SRCNN\\testsets\\Woman\\woman\\',i),new_img)
