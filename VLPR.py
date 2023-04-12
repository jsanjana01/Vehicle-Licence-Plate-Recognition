import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread("D:\\car1.jpeg") #it reads the image 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image from one color to another
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)) #display in plot

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #boundary points
contours = imutils.grab_contours(keypoints) #simplifies the contour points
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#to check wheather the contours form number plate or not
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True) #approx of a shape of contour
    if len(approx) == 4:
        location = approx
        break
        
location #location of number plate

mask = np.zeros(gray.shape, np.uint8) #creates a blank mask of the gray image
new_image = cv2.drawContours(mask, [location], 0,255, -1) #drawn contour on mask
new_image = cv2.bitwise_and(img, img, mask=mask) #mask overlaps on gray image

plt.imshow(new_image)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

(x,y) = np.where(mask==255) #finds points where image is black
(x1, y1) = (np.min(x), np.min(y)) 
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1] #cropped image is formed

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

reader = easyocr.Reader(['en']) #pass the language
result = reader.readtext(cropped_image) #reads the text
result

text = result[0][-2]
print(text)
font = cv2.FONT_HERSHEY_SIMPLEX #font of the text
#font specifications
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3) #draw a rectangle
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

#show from which state it belongs to
states={"AN":"Andaman and Nicobar","AP":"Andhra Pradesh","AR":"Arunachal Pradesh","AS":"Assam","BR":"Bihar","CH":"Chandigarh","DN":"Dadra and Nagar Haveli","DD":"Daman and Diu","DL":"Delhi","GA":"Goa","GJ":"Gujarat",
"HR":"Haryana","HP":"Himachal Pradesh","JK":"Jammu and Kashmir","KA":"Karnataka","KL":"Kerala","LD":"Lakshadweep","MP":"Madhya Pradesh","MH":"Maharashtra","MN":"Manipur","ML":"Meghalaya","MZ":"Mizoram","NL":"Nagaland","OD":"Odissa","PY":"Pondicherry","PN":"Punjab","RJ":"Rajasthan","SK":"Sikkim","TN":"TamilNadu","TR":"Tripura","UP":"Uttar Pradesh", "WB":"West Bengal","CG":"Chhattisgarh","TS":"Telangana","JH":"Jharkhand","UK":"Uttarakhand"}
stat = text[0:2]
try:
# Fetch the State information
    print('Car Belongs to',states[stat])
except:
    print('State not recognised!!')
