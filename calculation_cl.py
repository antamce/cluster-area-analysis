import tifffile as tiff
from imutils import contours
from skimage import measure
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def get_mask(img, threshold_value, dilatation_coef):
	blurred = cv2.GaussianBlur(img, (3, 3), 0)
	blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
	thresh = cv2.threshold(blurred, blurred.max()*threshold_value, blurred.max(), cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=dilatation_coef)
	thresh = np.where(thresh>0, 1, 0)
	labels = measure.label(thresh, connectivity=2)
	mask = np.zeros(thresh.shape, dtype="uint8")
	for label in np.unique(labels):
		if label == 0:
			continue
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		if numPixels > 30:
			mask = cv2.add(mask, labelMask)
	return mask

def get_mask_gfp(img, threshold_value, dilatation_coef):
	blurred = cv2.GaussianBlur(img, (3, 3), 0)
	blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
	thresh = cv2.threshold(blurred, blurred.max()*threshold_value, blurred.max(), cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=dilatation_coef)
	thresh = np.where(thresh>0, 1, 0)
	labels = measure.label(thresh, connectivity=2)
	mask = np.zeros(thresh.shape, dtype="uint8")
	for label in np.unique(labels):
		if label == 0:
			continue
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		if numPixels > 1000:
			mask = cv2.add(mask, labelMask)
	return mask

def gr_cnt(img, numb, tif, tif2, gfp_threshold, synpo_threshold, synpo_dil, gfp_dil=14): 

    mask = get_mask_gfp(img, gfp_threshold, gfp_dil)

    syn = tif2[numb]	
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    synpo_mask = cv2.bitwise_and(syn, syn, mask=mask.copy())
    binary_synpo = get_mask(synpo_mask, synpo_threshold, synpo_dil)
    cnts = imutils.grab_contours(cnts)
    colour_synpo = cv2.cvtColor(binary_synpo, cv2.COLOR_GRAY2BGR)
    colour_synpo[:,:,2] = 0
    colour_synpo[:,:,1] = 0
    cnts_synpo = cv2.findContours(binary_synpo.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_synpo = imutils.grab_contours(cnts_synpo)
    if bool(cnts):
        list_of_contours = []
        for i in range(len(cnts)):
            cv2.drawContours(colour_synpo , cnts, i, (20055,20055,20055), 3)
            cv2.drawContours(synpo_mask , cnts, i, (20055,20055,20055), 3)
    return cnts_synpo, colour_synpo, list_of_contours, synpo_mask

def textify(img, numb, list_of_contours, cnts_synpo, colour_synpo):
	for i, c in enumerate(cnts_synpo):
							M= cv2.moments(c)
							cx= int(M['m10']/M['m00'])
							cy= int(M['m01']/M['m00'])
							cv2.putText(colour_synpo, text= str(i+1), org=(cx,cy),
									fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(20055,20055,0),
									thickness=2, lineType=cv2.LINE_AA)
							cv2.putText(colour_synpo, text= str(numb+1), org=(30,30),
									fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,20055,20055),
									thickness=2, lineType=cv2.LINE_AA)
							list_of_contours.append(cv2.contourArea(c))
	return list_of_contours, colour_synpo
						
def calculation_cl(tif, tif2, gfp_threshold, synpo_threshold, synpo_dil, gfp_dil, sgfp, ssyn, ssynu, dfpath):
	
	
	df = pd.DataFrame()
	df['numbers'] = pd.Series([i for i in range(1, 180)])
	if isinstance(tif, str):
		tif = tiff.imread(tif)
		tif2 = tiff.imread(tif2)

	with tiff.TiffWriter(sgfp+'.tif') as stack:
		with tiff.TiffWriter(ssyn+'.tif') as stack2:
			with tiff.TiffWriter(ssynu+'.tif') as stack3:
				for numb, img in enumerate(tif): 
					cnts_synpo, colour_synpo, list_of_contours, synpo_mask = gr_cnt(img, numb, tif, tif2, gfp_threshold, synpo_threshold, synpo_dil)
							
					stack.write(colour_synpo, contiguous=True)
						
					list_of_contours, colour_synpo = textify(img, numb, list_of_contours, cnts_synpo, colour_synpo)
					df[f'{numb+1}'] = pd.Series(list_of_contours)
					stack2.write(synpo_mask, contiguous=True)
					stack3.write(colour_synpo, contiguous=True)


	df.to_excel(dfpath)