import numpy as np
import cv2

def meters_to_pixels(m, resx, resy):
    return max(1, int(round(m / ((resx + resy) / 2.0))))

def area_filter_bool(mask_bool, resx, resy, min_area_m2):
    if min_area_m2 <= 0:
        return mask_bool
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), 8)
    keep = np.zeros_like(mask_bool, bool)
    px_area = resx * resy
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] * px_area >= min_area_m2:
            keep[labels == i] = True
    return keep

def morph_open_bool(mask_bool, radius_m, resx, resy, iters=1):
    px = meters_to_pixels(radius_m, resx, resy)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    out = cv2.morphologyEx((mask_bool.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel, iterations=iters) > 0
    return out

def morph_close_bool(mask_bool, radius_m, resx, resy, iters=1):
    px = meters_to_pixels(radius_m, resx, resy)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    out = cv2.morphologyEx((mask_bool.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel, iterations=iters) > 0
    return out
