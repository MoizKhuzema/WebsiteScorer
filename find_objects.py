#http://stackoverflow.com/questions/28498831/opencv-get-centers-of-multiple-objects
    
import cv2
import numpy

def num_objects(img, thresh):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    centres = []
    for i in range(len(contours)):
      if cv2.contourArea(contours[i]) < thresh:
        continue
      moments = cv2.moments(contours[i])
      centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
      cv2.circle(img, centres[-1], 3, (0, 0, 0), -1)
    
    #cv2.imshow('image', img)
    #cv2.imwrite('output_2.png',img)
    #print len(centres)
    return len(centres)
    
