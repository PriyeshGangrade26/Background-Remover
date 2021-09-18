import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listImg = os.listdir("photo")
print(listImg)
imgList = []
for imgPath in listImg:
    img = cv2.imread(f"photo/{imgPath}")
    imgList.append(img)
print(len(imgList))

indexImg = 0
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,imgList[indexImg], threshold=0.8)
    imageStacked = cvzone.stackImages([img,imgOut],2,1)
    _, imageStacked = fpsReader.update(imageStacked)
    cv2.imshow("Image", imageStacked)
    key = cv2.waitKey(1)
    print(indexImg)
    if key == ord('a'):
        indexImg -=1
    elif key == ord('d'):
        indexImg +=1
    elif key == ord('q'):
        break

