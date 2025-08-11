import argparse
import numpy as np
import cv2

from config import (
    RGB_PATH, DSM_PATH, OUT_GPKG, OUT_LAYER,
    SMOOTH_RADIUS_M, REL_HEIGHT_THRESH, MIN_AREA_M2,
    VARI_MARGIN, OPENING_RADIUS_M, OPENING_ITERS,
    CLOSE_RADIUS_M, FILL_ALL_HOLES, MAX_HOLE_AREA_M2, FINAL_OPEN_RADIUS_M,
    SMOOTH_SIGMA_PX, SEED_PCT, ENERGY_SMOOTH_PX, MIN_SEED_AREA_M2, OPEN_ITERS_SEED,
    MIN_POLY_AREA_M2, MAX_POLY_AREA_M2, SIMPLIFY_TOL_M, POST_SIMPLIFY_FIX,
    FIX_INVALID, SAVE_GPKG
)

from utils.io_ops import save_gpkg
from utils.raster_ops import load_rgb, reproject_to_rgb_grid
from utils.masks import elevated_mask, low_vari_mask, finalize_mask
from utils.facets import normals_energy, seed_from_energy, watershed_per_cc
from utils.vectorize import polygonize_labels, postprocess_gdf


def parse_args():
    ap = argparse.ArgumentParser("Roof facet extraction")
    ap.add_argument("--rgb", default=RGB_PATH, help="Path to RGB GeoTIFF")
    ap.add_argument("--dsm", default=DSM_PATH, help="Path to DSM GeoTIFF")
    ap.add_argument("--out", default=OUT_GPKG, help="Output GeoPackage path")
    ap.add_argument("--layer", default=OUT_LAYER, help="Output layer name")
    return ap.parse_args()

def check_required_paths(args):
    missing = []
    if not args.rgb:
        missing.append("--rgb")
    if not args.dsm:
        missing.append("--dsm")
    if not args.out:
        missing.append("--out")
    if not args.layer:
        missing.append("--layer")

    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

def main():
    args = parse_args()
    check_required_paths(args)
    # Load rasters 
    print("Load rasters ")
    rgb01, rgb_transform, rgb_crs, H, W, resx, resy = load_rgb(args.rgb)
    dsm_f = reproject_to_rgb_grid(args.dsm, H, W, rgb_transform, rgb_crs)

    # Create masks
    print("Create masks ")
    elevated = elevated_mask(dsm_f, resx, resy, SMOOTH_RADIUS_M, REL_HEIGHT_THRESH, MIN_AREA_M2)
    low_var, thr, thr_eff = low_vari_mask(rgb01, VARI_MARGIN, OPENING_RADIUS_M, OPENING_ITERS, resx, resy, MIN_AREA_M2)
    final_mask = finalize_mask(
        elevated, low_var,
        CLOSE_RADIUS_M, FILL_ALL_HOLES, MAX_HOLE_AREA_M2,
        FINAL_OPEN_RADIUS_M, resx, resy, MIN_AREA_M2
    )
    final_bool = final_mask.astype(bool)

    print("Find flat planes ")
    # Facet energy
    E, E_masked = normals_energy(dsm_f, resx, resy, SMOOTH_SIGMA_PX, ENERGY_SMOOTH_PX, mask=final_bool)
    seed_bin, lab_seed, min_seed_px = seed_from_energy(E_masked, resx, resy, SEED_PCT, MIN_SEED_AREA_M2, open_iters=OPEN_ITERS_SEED)
    if seed_bin is None:
        print("No seeds found inside final mask. Exiting.")
        return
    print("Watershed in process ")
    # --- Watershed per CC 
    cc_refined, label_to_cc, _ = watershed_per_cc(final_bool, lab_seed, min_seed_px, E)

    # --- Vectorize ---
    gdf = polygonize_labels(cc_refined, (cc_refined > 0), rgb_transform, rgb_crs, label_to_cc)
    print("Postprocess polygons ")
    # --- Postprocess polygons ---
    gdf = postprocess_gdf(
        gdf,
        fill_holes=FILL_ALL_HOLES, max_hole_area_m2=MAX_HOLE_AREA_M2,
        simplify_tol_m=SIMPLIFY_TOL_M, post_simplify_fix=POST_SIMPLIFY_FIX,
        min_poly_area_m2=MIN_POLY_AREA_M2, max_poly_area_m2=MAX_POLY_AREA_M2,
        fix_invalid=FIX_INVALID
    )
    print("Save output ")
    # --- Save ---
    if SAVE_GPKG and not gdf.empty:
        save_gpkg(gdf, args.out, args.layer)
        print(f"✅ Saved {len(gdf)} polygons -> {args.out}:{args.layer}")
    else:
        print("⚠️ Nothing to save (empty GeoDataFrame).")
    print("Done.")
if __name__ == "__main__":
    main()
