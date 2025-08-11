import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import threshold_otsu
from .morpho import area_filter_bool, morph_open_bool, morph_close_bool, meters_to_pixels
import cv2

def elevated_mask(dsm_f, resx, resy, smooth_radius_m, rel_height_thresh, min_area_m2):
    sigma_px = smooth_radius_m / ((resx + resy) / 2.0)
    dsm_ground = gaussian_filter(dsm_f, sigma=sigma_px)
    rel_h = dsm_f - dsm_ground
    elevated = rel_h > rel_height_thresh
    elevated = area_filter_bool(elevated, resx, resy, min_area_m2)
    return elevated

def low_vari_mask(rgb01, vari_margin, opening_radius_m, opening_iters, resx, resy, min_area_m2):
    R, G, B = rgb01[...,0], rgb01[...,1], rgb01[...,2]
    den = R + G - B
    den = np.where(np.abs(den) < 1e-6, 1e-6, den)
    VARI = (G - R) / den
    vals = VARI[np.isfinite(VARI)]
    thr = threshold_otsu(np.clip(vals, -1, 1)) if vals.size else 0.0
    thr_eff = thr - vari_margin
    low_vari = (VARI < thr_eff) & np.isfinite(VARI)

    low_vari_open = morph_open_bool(low_vari, opening_radius_m, resx, resy, iters=opening_iters)
    low_vari_open = area_filter_bool(low_vari_open, resx, resy, min_area_m2)
    return low_vari_open, thr, thr_eff

def finalize_mask(elevated, low_vari_open, close_radius_m, fill_all_holes, max_hole_area_m2,
                  final_open_radius_m, resx, resy, min_area_m2):
    final_mask = elevated & low_vari_open
    final_mask = area_filter_bool(final_mask, resx, resy, min_area_m2)

    # Closing small breaks
    final_mask = morph_close_bool(final_mask, close_radius_m, resx, resy, iters=1)

    # Fill holes
    if fill_all_holes:
        final_mask = binary_fill_holes(final_mask)
    else:
        from skimage.morphology import remove_small_holes
        area_thresh_px = int(np.ceil(max_hole_area_m2 / (resx*resy)))
        final_mask = remove_small_holes(final_mask, area_threshold=area_thresh_px)

    # Light opening to smooth spurs
    final_mask = morph_open_bool(final_mask, final_open_radius_m, resx, resy, iters=1)
    return final_mask
