import cv2
import imutils
from skimage.transform import resize

img = cv2.imread('data/test/(183).jpg')
h , w , c = img.shape
resized = cv2.resize(img , (100,100) , cv2.INTER_LINEAR)
img_retornar = cv2.resize(resized , ( w , h) , cv2.INTER_LINEAR)
print(img.dtype)
print(img_retornar.dtype)
fim = cv2.addWeighted(img , 0.7,img_retornar , 0.3 , 0)
'''


'''

cv2.imshow("test" , img)
cv2.imshow("100" ,resized )
cv2.imshow("Resize" ,img_retornar )
cv2.imshow("fim" ,fim  )
cv2.waitKey(0)
'''


