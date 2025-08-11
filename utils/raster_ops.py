import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def load_rgb(path):
    with rasterio.open(path) as r:
        rgb = r.read().astype(np.float32)
        transform = r.transform
        crs = r.crs
    rgb = rgb.transpose(1,2,0)  # (H,W,C)
    rgb01 = rgb / (65535.0 if rgb.max() > 255 else 255.0)
    rgb01 = np.clip(rgb01, 0, 1)
    H, W, _ = rgb01.shape
    resx, resy = abs(transform.a), abs(transform.e)
    return rgb01, transform, crs, H, W, resx, resy

def reproject_to_rgb_grid(dsm_path, H, W, rgb_transform, rgb_crs):
    with rasterio.open(dsm_path) as s:
        dsm = np.full((H, W), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(s, 1),
            destination=dsm,
            src_transform=s.transform, src_crs=s.crs,
            dst_transform=rgb_transform, dst_crs=rgb_crs,
            resampling=Resampling.bilinear,
        )
    valid = np.isfinite(dsm)
    fill_val = np.nanmedian(dsm[valid]) if valid.any() else 0.0
    dsm_f = dsm.copy()
    dsm_f[~valid] = fill_val
    return dsm_f
