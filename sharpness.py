import numpy as np
import cv2


def sharpness_factor(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(grayscale)
    
    img = cl1
    array = np.asarray(img, dtype=np.int32)
    
    #gy, gx = np.gradient(array)
    #gnorm = np.sqrt(gx**2 + gy**2)
    #sharpness = np.average(gnorm)
    #print "Image Sharpness: ", sharpness
    
    dx = np.diff(array)[1:,:] # remove the first row
    dy = np.diff(array, axis=0)[:,1:] # remove the first column
    dnorm = np.sqrt(dx**2 + dy**2)
    sharpness = np.average(dnorm)
    return sharpness
    #print "Image Sharpness: ", sharpness
#print sharpness_factor("Inputs\input18.png")

#print sharpness_factor("Inputs\input3.png")