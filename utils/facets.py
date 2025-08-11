import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, sobel
from skimage.segmentation import watershed

def normals_energy(dsm_f, resx, resy, smooth_sigma_px, energy_smooth_px, mask=None):
    dsm_s = gaussian_filter(dsm_f, sigma=smooth_sigma_px)
    dzdy, dzdx = np.gradient(dsm_s, resy, resx)

    Nx, Ny, Nz = -dzdx, -dzdy, np.ones_like(dzdx)
    L = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz) + 1e-9
    Nx /= L; Ny /= L; Nz /= L

    gx1, gy1 = sobel(Nx, axis=1)/8.0, sobel(Nx, axis=0)/8.0
    gx2, gy2 = sobel(Ny, axis=1)/8.0, sobel(Ny, axis=0)/8.0
    gx3, gy3 = sobel(Nz, axis=1)/8.0, sobel(Nz, axis=0)/8.0
    E = (gx1*gx1 + gy1*gy1 + gx2*gx2 + gy2*gy2 + gx3*gx3 + gy3*gy3) ** 0.5
    E = gaussian_filter(E.astype(np.float32), sigma=energy_smooth_px)

    if mask is not None:
        E_masked = E.copy()
        E_masked[~mask] = np.nan
    else:
        E_masked = E
    return E, E_masked

def seed_from_energy(E_masked, resx, resy, seed_pct, min_seed_area_m2, open_iters=1):
    vals = E_masked[np.isfinite(E_masked)]
    if vals.size == 0:
        return None, None, None
    thr_seed = np.percentile(vals, seed_pct)
    seed_bin = (E_masked <= thr_seed)
    # area filter
    px_area = float(resx * resy)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(seed_bin.astype(np.uint8), 8)
    min_seed_px = max(1, int(np.ceil(min_seed_area_m2 / px_area)))
    valid_seed = np.zeros(num, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_seed_px:
            valid_seed[i] = True
    seed_bin = valid_seed[lab]

    if open_iters > 0:
        seed_bin = cv2.morphologyEx(seed_bin.astype(np.uint8),
                                    cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),
                                    iterations=open_iters) > 0
    num_seed, lab_seed = cv2.connectedComponents(seed_bin.astype(np.uint8), connectivity=8)
    return seed_bin, lab_seed, min_seed_px

def watershed_per_cc(final_bool, lab_seed, min_seed_px, E):
    H, W = final_bool.shape
    num_cc, lab_cc, stats_cc, _ = cv2.connectedComponentsWithStats(final_bool.astype(np.uint8), connectivity=8)
    cc_refined   = np.zeros((H, W), dtype=np.int32)
    label_to_cc  = {}
    next_label   = 1

    for cc_id in range(1, num_cc):
        comp_mask = (lab_cc == cc_id)
        inter_seed = np.unique(lab_seed[comp_mask])
        inter_seed = inter_seed[inter_seed != 0]

        good_seeds = []
        for s_lbl in inter_seed:
            seed_in_cc = (lab_seed == s_lbl) & comp_mask
            if seed_in_cc.sum() >= min_seed_px:
                good_seeds.append(s_lbl)

        if len(good_seeds) <= 1:
            cc_refined[comp_mask] = next_label
            label_to_cc[next_label] = cc_id
            next_label += 1
            continue

        markers_local = np.zeros((H, W), dtype=np.int32)
        for i, s_lbl in enumerate(good_seeds, start=1):
            markers_local[(lab_seed == s_lbl) & comp_mask] = i

        ws_labels = watershed(E, markers=markers_local, mask=comp_mask)
        for i in range(1, len(good_seeds) + 1):
            m = (ws_labels == i)
            cc_refined[m] = next_label
            label_to_cc[next_label] = cc_id
            next_label += 1

    final_mask_refined = (cc_refined > 0)
    return cc_refined, label_to_cc, final_mask_refined
