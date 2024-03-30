 # -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:06:40 2016
Collected and modified from all over the internet by Erkka Virtanen

Thanks to:
    
CharlesLeifer
http://charlesleifer.com/blog/using-python-and-k-means-to-find-the-dominant-colors-in-images/

Creators of SalientDetect:
https://github.com/NLeSC/SalientDetector-python

texasflood
http://stackoverflow.com/questions/28498831/opencv-get-centers-of-multiple-objects

Bikramjot Singh Hanzra 
http://hanzratech.in/2015/01/16/color-difference-between-2-colors-using-python.html

    
HOW TO USE:
The script takes one picture-file as the argument and outputs a score of x/99
of how perfect the website is aesthetically, 99 being perfect and 0 being unusable.
The script also outputs the reasons for the score.
Use 1 as the second argument to print the rating and reasoning on to the image.
Otherwise use 0.
    
"""

#IMPORT ALL SCRIPTS
import cv2
import salient_regions
import find_objects
import color_rating
import aescript2
import sharpness


def predict(image):    
    #print name
    stars = 0
    #minimum threshold for salient regions
    arg_t = 500
    #nestedness lower threshold
    arg3 = 0.5
    #nestedness upper threshold
    arg4 = 0.8
    #Font used, background
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #Run all scripts
    salient_regions.find_salient(image)
    objects = find_objects.num_objects(image, arg_t)
    color_distance = color_rating.rate(image)
    sharp_factor = sharpness.sharpness_factor(image)
    nestedness= aescript2.main(image,arg3,arg4)
    
    #Give a rating to based on the results of the scripts
    #Number of objects
    points1 = 0
    remarks = []
    #print objects
    if 14 <= objects <= 19:
        points1 = 33
        remarks.append("-Perfect amount of items on the page!")
    elif 8 <= objects < 14:
        points1 = 30
        remarks.append("-Good amount of items on the page")
    elif 4 <= objects < 8:
        points1 = 25
        remarks.append("-Not enough points of interest on page.")
    elif objects <= 4:
        points1 = 10
        remarks.append("-Page has too many large connected areas.")
    elif 20 <= objects < 26:
        points1 = 30
        remarks.append("-Good amount of items on the page.")
    elif 26 <= objects <= 40:
        points1 = 22
        remarks.append("-Slightly too many items on the page.")
    elif 40 < objects < 60:
        points1 = 13
        remarks.append("-Too many items on the page.")
    elif objects >= 60:
        points1 = 1
        remarks.append("-WAY too many items on the page.")
    
    #Color distance
    points2 = 0
    #print color_distance
    if 68 <= color_distance <= 85:
        points2 = 18
        remarks.append("-Colors are chosen perfectly!")
    elif 55 <= color_distance < 68:
        points2 = 15
        remarks.append("-Color range is good.")
    elif 45 <= color_distance < 68:
        points2 = 10
        remarks.append("-Colors are spread little too far apart.")
    elif color_distance < 45:
        points2 = 5
        remarks.append("-Color range is too shallow.")
    elif 85 <= color_distance < 95:
        points2 = 15
        remarks.append("-Colors range is good.")
    elif 95 < color_distance <= 120:
        points2 = 10
        remarks.append("-Colors are spread little too far apart.")
    elif color_distance > 120:
        points2 = 5
        remarks.append("-Color range is too large.")
    
    #Sharpness Factor
    points_s = 0
    if 11 <= sharp_factor <= 19:
        points_s = 15
        remarks.append("-Page looks perfectly sharp and in focus!")
    elif 6 <= sharp_factor < 11:
        points_s = 10
        remarks.append("-Page looks slightly soft.")
    elif sharp_factor < 6:
        points_s = 5
        remarks.append("-Page looks WAY too unfocused.")
    elif 19 < sharp_factor < 26:
        points_s = 10
        remarks.append("-Page looks a bit too sharp.")
    elif sharp_factor >= 26:
        points_s = 5
        remarks.append("-Page looks WAY too sharp!")
        
        
    #Nestedness level
    points3 = 0
    #print nestedness
    if 8000 <= nestedness <= 12000:
        points3 = 33
        remarks.append("-Perfect amount of order on the page!")
    elif 6000 <= nestedness < 8000:
        points3 = 30
        remarks.append("-Well structured page.")
    elif 2000 <= nestedness < 6000:
        points3 = 25
        remarks.append("-Page could be a bit more structured.")
    elif nestedness < 2000:
        points3 = 10
        remarks.append("-Page doesn't have enough structure.")
    elif 12000 <= nestedness < 14000:
        points3 = 30
        remarks.append("-Page structure is good.")
    elif 14000 < nestedness <= 20000:
        points3 = 23
        remarks.append("-Page is a little too rigid-looking.")
    elif nestedness > 20000:
        points3 = 5
        remarks.append("-Page has too much hierarchy.")
    
    #Calculate final score
    tikst = "*Points for Clutteredness: "+str(points1)+"/ 33"
    # printer(img,tikst,10,180,192,192,192,arg2)
    tikst = "*Points for Color Range: "+ str(points2) +"/ 18"
    # printer(img,tikst,10,210,192,192,192,arg2)
    tikst =  "*Points for Sharpness: "+ str(points_s) + "/ 15"
    # printer(img,tikst,10,240,192,192,192,arg2)
    tikst =  "*Points for Hierarchy: "+ str(points3) +"/ 33 \n"
    # printer(img,tikst,10,270,192,192,192,arg2)
    overall = points1+points2+points_s+points3
    tikst = "Overal Score: "+ str(overall)+ " / 99 points"
    # printer(img,tikst,10,300,192,192,192,arg2)
    if overall >= 89:
        remarks.append("A Very Good Looking Website! :)")
        # printer2(img,"*****",10,400,0,255,255,arg2)
        # fil = "Outputs\\5-Star\\"
        # dest = fil + name
        # cv2.imwrite(dest,img)
        stars = 5
    elif 79 <= overall < 89:
        remarks.append("A Good Looking Website!")
        # printer2(img,"****",10,400,0,255,255,arg2)
        # fil = "Outputs\\4-Star\\"
        # dest = fil + name
        # cv2.imwrite(dest,img)
        stars = 4
    elif 69 <= overall < 79:
        remarks.append("An Average Looking Website!")
        # printer2(img,"***",10,400,0,255,255,arg2)
        # fil = "Outputs\\3-Star\\"
        # dest = fil + name
        # cv2.imwrite(dest,img)
        stars = 3
    elif 59 <= overall < 69:
        remarks.append("A Below Average Looking Website!")
        # printer2(img,"**",10,400,0,255,255,arg2)
        # fil = "Outputs\\2-Star\\"
        # dest = fil + name
        # cv2.imwrite(dest,img)
        stars = 2
    elif overall < 59:
        remarks.append("A Poor Looking Website! :(")
        # printer2(img,"*",10,400,0,255,255,arg2)
        # fil = "Outputs\\1-Star\\"
        # dest = fil + name
        # cv2.imwrite(dest,img)
        stars = 1

    max_points = sharp_factor + nestedness + color_distance + objects
    return overall, max_points
'''
def runner(nimi):  
    result = ""
    score2 = main(nimi)
    for d in range(0,score2):
        result += "*"
    print "SCORE:", result
'''

'''
    
def ultra_runner(number_of_files):
    #number_of_files = 2
    numba = 0
    beginning = "Inputs\\input"
    end = ".png"
    filu = beginning + str(numba) + end
    for i in range(0,number_of_files):
        numba = numba+1
        filu = beginning + str(numba) + end
        runner(filu)
                         
'''                 
#ultra_runner(13)
# predict(r"C:\Users\moizk\Documents\Capture.png")
