import numpy as np
import os
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern


def color_hist(img,bins=10):
    # img must be numpy array in RGB for color Histogram
    hist = []
    for i in range(3): #for each channel(R,G,B)
        channel_hist = np.histogram(img[:,:,i],bins=bins,range=(0,1))[0]
        hist.extend(channel_hist)
    # we first convert histogram to np array and then divide by sum of all bins of the histogram
    hist = np.array(hist) / np.sum(hist)  # normalzie
    return hist

# for 10 bins 3 channel  ==> 30 diminsion features map




def edgg_features(img):
    # as img in srmalized(0-1) we convert it back to (0-255) and mek it 8 bit integer as open cv require it
    # and then we conver RGB-> Gray(as edge detection work on intensity change )
    gray  = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)  

    # Sobel -> Sobel operator is a gradient filter it detects edges by measuring change in pixel intensity.
    # Sobel(img,output_type(64bitFP),derivative_order,kernel_size(sobel filter))
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3) # edges in horizontal direction
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3) # edges in vertical direction
    
    #Gradient magnitude combining horizontal + vertical gradient)
    # Pythagoras theorem in 2D gradient space->sqrt(Gx^2 + Gy^2)
    # as Gx and Gy can be negative we use sqrt
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)

    # flatten
    # mean give -> average edge intensity
    # sd give  -> variation of edge intensity
    return np.array([np.mean(sobel_mag),np.std(sobel_mag)])



# LBP code → value between 0 and max code (P+2 for uniform method)
# for P=8 value in 2D array of lps is between (0,8+2)->(0,10)

def texture_features(img,P=8,R=1):
    # similar like above
    gray = cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    # we extract local binary patterent using skimage built in funciton 
    # methood = "uniform" -> it is used to reproduce the number of bins
    # lps -> is a 2D array of same size as gray image, with LBP codes per pixel.
    lps = local_binary_pattern(gray,P,R,method="uniform")
    
    # bin size = P+2 -> 8+2 -> 10  so each histgram cover all possible code of lps
    hist,_ = np.histogram(lps,bins=int(P+2),range=(0,P+2))
    # Divide each bin by total number of pixels
    hist = hist / hist.sum() #normalize
    return hist



# we combine all the 3 features
def extract_features(img):
    ch = color_hist(img)
    ef = edgg_features(img)
    tf = texture_features(img)
    return np.concatenate([ch,ef,tf])  # concatinating all 2 features



def extract_features_for_pipeline(x):
     # X is a list or array of images (64x64x3)
    features_list = []
    # we conver flatten images back to (64,64,3)
    for img in x:
        # if img is flattened then we reshape reshape
        if img.ndim == 1:
            img = img.reshape(64, 64, 3)

        features = extract_features(img)  #otherwise we img to as it is
        features_list.append(features)
    return np.array(features_list)


# feature_arr = extract_features_for_pipeline(np_x)
# print(feature_arr.shape)