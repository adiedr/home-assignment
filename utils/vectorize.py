import geopandas as gpd
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from rasterio.features import shapes

def polygonize_labels(cc_refined, mask, transform, crs, label_to_cc):
    records = []
    for geom, val in shapes(cc_refined.astype(np.int32),
                            mask=mask.astype(np.uint8),
                            transform=transform):
        val = int(val)
        if val == 0:
            continue
        records.append({
            "label_id": val,
            "orig_cc": label_to_cc.get(val, -1),
            "geometry": shape(geom)
        })
    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf

def fill_small_holes_geom(geom, max_hole_area_m2):
    if geom.is_empty or geom.area == 0:
        return geom

    def _fix_poly(p: Polygon) -> Polygon:
        kept = []
        for ring in p.interiors:
            hole = Polygon(ring)
            if hole.area > max_hole_area_m2:
                kept.append(ring)
        try:
            return Polygon(p.exterior, holes=kept)
        except Exception:
            return p.buffer(0)

    if isinstance(geom, Polygon):
        return _fix_poly(geom)
    elif isinstance(geom, MultiPolygon):
        parts = [_fix_poly(p) for p in geom.geoms]
        parts = [p.buffer(0) for p in parts if not p.is_empty]
        if not parts:
            return geom
        try:
            return MultiPolygon(parts)
        except Exception:
            return unary_union(parts)
    else:
        return geom

def postprocess_gdf(gdf, fill_holes, max_hole_area_m2,
                    simplify_tol_m, post_simplify_fix,
                    min_poly_area_m2, max_poly_area_m2, fix_invalid):
    if gdf.empty:
        return gdf

    # work in metric CRS for area/simplify
    if not gdf.crs or not gdf.crs.is_projected:
        utm = gdf.estimate_utm_crs()
        gdf_m = gdf.to_crs(utm)
    else:
        gdf_m = gdf.copy()

    if fill_holes:
        gdf_m["geometry"] = gdf_m["geometry"].apply(lambda g: fill_small_holes_geom(g, max_hole_area_m2))

    if simplify_tol_m and simplify_tol_m > 0:
        def _simplify(geom):
            s = geom.simplify(simplify_tol_m, preserve_topology=True)
            if post_simplify_fix:
                s = s.buffer(0)
            return s if (s and not s.is_empty) else geom
        gdf_m["geometry"] = gdf_m.geometry.apply(_simplify)
        gdf_m = gdf_m[gdf_m.geometry.notna() & ~gdf_m.geometry.is_empty].copy()

    gdf_m["area_m2"] = gdf_m.geometry.area
    keep = (gdf_m["area_m2"] >= min_poly_area_m2) & (gdf_m["area_m2"] <= max_poly_area_m2)
    gdf_m = gdf_m.loc[keep].copy()

    gdf_out = gdf_m.to_crs(gdf.crs) if gdf.crs else gdf_m

    if fix_invalid and not gdf_out.empty:
        gdf_out["geometry"] = gdf_out["geometry"].buffer(0)
        gdf_out = gdf_out[gdf_out.geometry.notna() & ~gdf_out.geometry.is_empty].copy()

    return gdf_out
