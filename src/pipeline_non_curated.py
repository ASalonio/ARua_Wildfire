"""
Non-Curated Wildfire Pipeline (Minimal QA)
==========================================
Minimal QA pipeline — the quick-and-dirty approach.

Only applies:
  - Valid data mask (both bands > 0, finite, not nodata)

Skips:
  - Cloud / smoke mask
  - Atmospheric correction QA
  - Saturation / outlier / edge masks
  - Vegetation filter (NDVI)
  - Brightness filter
  - Topographic correction (SCS+C)

Run from: c:\\Repos\\ARua_Wildfire\\
    python src/pipeline_non_curated.py
"""

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes, rasterize
from scipy.ndimage import binary_fill_holes, binary_closing
from shapely.geometry import shape
from shapely.ops import unary_union


# ═════════════════════════════════════════════════════════════
# DIRECTORIES & PATHS
# ═════════════════════════════════════════════════════════════

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "non_qa")

DATA_PREFIRE_DIR = os.path.join(DATA_DIR, "prefire")
DATA_POSTFIRE_DIR = os.path.join(DATA_DIR, "postfire")
OUT_PREFIRE_DIR = os.path.join(OUTPUT_DIR, "prefire")
OUT_POSTFIRE_DIR = os.path.join(OUTPUT_DIR, "postfire")

AOI_PATH = os.path.join(DATA_DIR, "aoi.geojson")
LANDUSE_PATH = os.path.join(DATA_DIR, "landuse_aoi.geojson")
FIRE_PATH = os.path.join(DATA_DIR, "true_perimeter.geojson")

clip_pre = {
    "B8A": os.path.join(DATA_PREFIRE_DIR, "clip_nir.tiff"),
    "B12": os.path.join(DATA_PREFIRE_DIR, "clip_swir2.tiff"),
}

clip_post = {
    "B8A": os.path.join(DATA_POSTFIRE_DIR, "clip_nir.tiff"),
    "B12": os.path.join(DATA_POSTFIRE_DIR, "clip_swir2.tiff"),
}

NBR_PRE = os.path.join(OUT_PREFIRE_DIR, "nbr_pre_no_qa.tif")
NBR_POST = os.path.join(OUT_POSTFIRE_DIR, "nbr_post_no_qa.tif")
DNBR_PATH = os.path.join(OUTPUT_DIR, "dnbr.tif")
SEVERITY_PATH = os.path.join(OUTPUT_DIR, "severity_no_qa.tif")
POLYGON_PATH = os.path.join(OUTPUT_DIR, "burn_polygons_no_qa.gpkg")


# ═════════════════════════════════════════════════════════════
# NBR
# ═════════════════════════════════════════════════════════════

def non_curated_nbr(b8a_path, b12_path, out_path=None):
    """
    Calculate basic NBR (no QA, no brightness filter).

    Uses only B8A (NIR narrow) and B12 (SWIR2).
    Allows small negative reflectance (common in S2 L2A clips).
    """
    print("\nCalculating NBR (non-curated / no QA)...")

    with rasterio.open(b8a_path) as src:
        b8a = src.read(1).astype("float32")
        meta = src.meta.copy()
        nodata = src.nodata if src.nodata is not None else -9999.0

    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype("float32")

    print(f"  B8A min/mean/max: {b8a.min():.4f} / {b8a.mean():.4f} / {b8a.max():.4f}")
    print(f"  B12 min/mean/max: {b12.min():.4f} / {b12.mean():.4f} / {b12.max():.4f}")

    eps = 1e-10

    data_exists = (
        (b8a != nodata) & np.isfinite(b8a) &
        (b12 != nodata) & np.isfinite(b12)
    )
    denom_ok = (b8a + b12) > -0.15
    compute_mask = data_exists & denom_ok

    nodata_out = -9999.0
    nbr = np.full_like(b8a, nodata_out, dtype="float32")
    nbr[compute_mask] = (
        (b8a[compute_mask] - b12[compute_mask])
        / (b8a[compute_mask] + b12[compute_mask] + eps)
    )

    # Clip to physical range
    nbr_valid = nbr[compute_mask]
    out_of_range = np.sum((nbr_valid < -1) | (nbr_valid > 1))
    if out_of_range > 0:
        print(f"  Clipped {out_of_range:,} NBR values outside [-1, 1]")
        nbr[compute_mask] = np.clip(nbr[compute_mask], -1.0, 1.0)

    masked_count = np.sum(~compute_mask)
    print(f"  Masked pixels: {masked_count:,} ({masked_count / nbr.size:.1%})")
    print(f"  Valid pixels:  {np.sum(compute_mask):,} ({np.mean(compute_mask):.1%})")

    meta.update(driver="GTiff", count=1, dtype="float32",
                nodata=nodata_out, compress="lzw")

    if out_path:
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(nbr, 1)

    return nbr


# ═════════════════════════════════════════════════════════════
# dNBR
# ═════════════════════════════════════════════════════════════

def calculate_and_save_dnbr(nbr_pre_path, nbr_post_path, output_path):
    """Calculate dNBR = NBR_pre - NBR_post."""

    with rasterio.open(nbr_pre_path) as src:
        nbr_pre = src.read(1)
        profile = src.profile.copy()
        nodata_in = src.nodata if src.nodata is not None else -9999.0

    with rasterio.open(nbr_post_path) as src:
        nbr_post = src.read(1)

    valid_mask = (nbr_pre != nodata_in) & (nbr_post != nodata_in)

    dnbr = np.full_like(nbr_pre, nodata_in, dtype=np.float32)
    dnbr[valid_mask] = nbr_pre[valid_mask] - nbr_post[valid_mask]

    dnbr_valid = dnbr[valid_mask]
    print(f"\n  Valid pixels: {valid_mask.sum():,} ({valid_mask.sum()/dnbr.size*100:.1f}%)")
    print(f"  dNBR range: {dnbr_valid.min():.3f} to {dnbr_valid.max():.3f}")

    out_of_range = np.sum((dnbr_valid < -2) | (dnbr_valid > 2))
    if out_of_range > 0:
        print(f"  Warning: {out_of_range} dNBR values outside [-2, 2]")

    profile.update(dtype="float32", nodata=-9999.0, compress="lzw")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dnbr, 1)

    print(f"  dNBR saved to: {output_path}")
    return dnbr


# ═════════════════════════════════════════════════════════════
# SEVERITY CLASSIFICATION
# ═════════════════════════════════════════════════════════════

SEVERITY_LABELS = {
    0: "Unburned",
    1: "Low",
    2: "Moderate-Low",
    3: "Moderate-High",
    4: "High",
}


def classify_severity(dnbr_path, output_path):
    """
    USGS severity classes from dNBR.
    0=Unburned, 1=Low, 2=Moderate-Low, 3=Moderate-High, 4=High, 255=nodata.
    """
    with rasterio.open(dnbr_path) as src:
        dnbr = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999.0

    sev = np.full_like(dnbr, 255, dtype=np.uint8)
    valid = dnbr != nodata

    sev[valid & (dnbr < 0.10)] = 0
    sev[valid & (dnbr >= 0.10) & (dnbr < 0.27)] = 1
    sev[valid & (dnbr >= 0.27) & (dnbr < 0.44)] = 2
    sev[valid & (dnbr >= 0.44) & (dnbr < 0.66)] = 3
    sev[valid & (dnbr >= 0.66)] = 4

    profile.update(dtype="uint8", nodata=255, compress="lzw", count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(sev, 1)

    print(f"\n  Severity saved to: {output_path}")
    return sev


# ═════════════════════════════════════════════════════════════
# BURN SCAR POLYGONS
# ═════════════════════════════════════════════════════════════

def create_burn_scar_polygon(dnbr_path, threshold, min_area_ha,
                              output_path=None):
    """
    Convert dNBR raster to burn scar polygon(s).

    Steps: threshold → fill holes → close gaps → vectorize → filter by area → dissolve.
    """
    with rasterio.open(dnbr_path) as src:
        dnbr = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else -9999.0

    burn_mask = np.zeros_like(dnbr, dtype="uint8")
    valid_mask = dnbr != nodata
    burn_mask[valid_mask & (dnbr >= threshold)] = 1

    burned_px = np.sum(burn_mask == 1)
    total_valid = np.sum(valid_mask)
    print(f"\n  Threshold: dNBR >= {threshold}")
    print(f"  Burned pixels: {burned_px:,} ({burned_px / total_valid * 100:.2f}%)")

    # Morphology
    burn_mask = binary_fill_holes(burn_mask).astype("uint8")
    burn_mask = binary_closing(burn_mask, structure=np.ones((3, 3))).astype("uint8")

    burn_px_after = np.sum(burn_mask == 1)
    print(f"  After morphology: {burn_px_after:,} ({burn_px_after / total_valid * 100:.2f}%)")

    # Vectorize
    records = []
    for geom, value in shapes(burn_mask, mask=(burn_mask == 1), transform=transform):
        records.append({"geometry": shape(geom), "burned": int(value)})

    if not records:
        print("  No burned areas detected above threshold")
        return None

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf["area_ha"] = gdf.geometry.area / 10_000

    print(f"  Initial polygons: {len(gdf)}")
    print(f"  Total area: {gdf['area_ha'].sum():.2f} ha")

    gdf = gdf[gdf["area_ha"] >= min_area_ha].copy()

    # Dissolve
    merged = unary_union(gdf.geometry)
    gdf_out = gpd.GeoDataFrame(
        {
            "burned": [1],
            "area_ha": [merged.area / 10_000],
            "area_km2": [merged.area / 1_000_000],
            "perimeter_km": [merged.length / 1000],
            "n_patches": [len(gdf)],
            "dnbr_threshold": [threshold],
        },
        geometry=[merged],
        crs=crs,
    )

    print(f"  Final area: {gdf_out['area_ha'].iloc[0]:.2f} ha")

    if output_path:
        gdf_out.to_file(output_path, driver="GPKG")
        print(f"  Saved to: {output_path}")

    return gdf_out


# ═════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════

def compare_burn_scars(predicted_path, reference_path, output_dir):
    """
    Compare predicted burn scar with reference (e.g. Copernicus EMS).
    Saves error polygons (.gpkg) and metrics (.csv).
    Returns dict with all metrics.
    """
    pred_gdf = gpd.read_file(predicted_path)
    ref_gdf = gpd.read_file(reference_path)

    if pred_gdf.crs != ref_gdf.crs:
        ref_gdf = ref_gdf.to_crs(pred_gdf.crs)

    pred_geom = unary_union(pred_gdf.geometry)
    ref_geom = unary_union(ref_gdf.geometry)

    pred_area = pred_geom.area
    ref_area = ref_geom.area
    pred_perim = pred_geom.length
    ref_perim = ref_geom.length

    intersection = pred_geom.intersection(ref_geom)
    union = pred_geom.union(ref_geom)
    fp = pred_geom.difference(ref_geom)
    fn = ref_geom.difference(pred_geom)

    iou = intersection.area / union.area
    dice = (2 * intersection.area) / (pred_area + ref_area)
    precision = intersection.area / pred_area if pred_area > 0 else 0
    recall = intersection.area / ref_area if ref_area > 0 else 0

    area_diff_pct = (pred_area - ref_area) / ref_area * 100
    perim_diff_pct = (pred_perim - ref_perim) / ref_perim * 100

    print(f"\n{'=' * 60}")
    print("BURN SCAR COMPARISON (non-curated)")
    print(f"{'=' * 60}")
    print(f"  Reference area:  {ref_area/10_000:>10,.2f} ha")
    print(f"  Predicted area:  {pred_area/10_000:>10,.2f} ha ({area_diff_pct:+.2f}%)")
    print(f"  IoU:             {iou:.4f} ({iou*100:.2f}%)")
    print(f"  Dice/F1:         {dice:.4f} ({dice*100:.2f}%)")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  False Pos:       {fp.area/10_000:,.2f} ha")
    print(f"  False Neg:       {fn.area/10_000:,.2f} ha")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    error_gdf = gpd.GeoDataFrame(
        {
            "error_type": ["False Positive", "False Negative"],
            "area_ha": [fp.area / 10_000, fn.area / 10_000],
        },
        geometry=[fp, fn],
        crs=pred_gdf.crs,
    )
    error_gdf.to_file(os.path.join(output_dir, "burn_scar_errors.gpkg"), driver="GPKG")

    metrics = {
        "area_pred_ha": pred_area / 10_000,
        "area_ref_ha": ref_area / 10_000,
        "area_diff_pct": area_diff_pct,
        "perimeter_pred_km": pred_perim / 1000,
        "perimeter_ref_km": ref_perim / 1000,
        "perimeter_diff_pct": perim_diff_pct,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "false_positive_ha": fp.area / 10_000,
        "false_negative_ha": fn.area / 10_000,
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "comparison_metrics.csv"), index=False
    )

    return metrics


# ═════════════════════════════════════════════════════════════
# SEVERITY STATISTICS
# ═════════════════════════════════════════════════════════════

def severity_area(predicted_polygon_path, severity_path, output_dir,
                  severity_labels=None):
    """Area by severity class inside predicted burn perimeter."""

    if severity_labels is None:
        severity_labels = SEVERITY_LABELS

    pred_gdf = gpd.read_file(predicted_polygon_path)
    if pred_gdf.crs != "EPSG:32629":
        pred_gdf = pred_gdf.to_crs("EPSG:32629")

    pred_geom = unary_union(pred_gdf.geometry)
    shapes_geom = [pred_geom.__geo_interface__]

    with rasterio.open(severity_path) as src:
        sev_data, sev_transform = mask(src, shapes_geom, crop=True,
                                        nodata=255, filled=False)
        sev = sev_data[0]

    sev_flat = sev[~sev.mask]
    if sev_flat.size == 0:
        print("  No valid pixels inside predicted area.")
        return None

    pixel_area_ha = abs(sev_transform.a * sev_transform.e) / 10_000
    unique, counts = np.unique(sev_flat, return_counts=True)

    rows = []
    for val, cnt in zip(unique, counts):
        if int(val) not in severity_labels:
            continue
        rows.append({
            "severity": int(val),
            "severity_label": severity_labels[int(val)],
            "area_ha": cnt * pixel_area_ha,
        })

    df = pd.DataFrame(rows).sort_values("severity")
    df.to_csv(os.path.join(output_dir, "severity_aoi_non_qa.csv"), index=False)

    print(f"\n  Severity area breakdown:")
    print(df.to_string(index=False))
    return df


def severity_landcover_stats(predicted_polygon_path, severity_path,
                              landcover_path, output_dir,
                              severity_labels=None,
                              landcover_class_col="info"):
    """Cross-tabulate severity × land cover inside predicted burn perimeter."""

    if severity_labels is None:
        severity_labels = SEVERITY_LABELS

    pred_gdf = gpd.read_file(predicted_polygon_path)
    if pred_gdf.crs != "EPSG:32629":
        pred_gdf = pred_gdf.to_crs("EPSG:32629")
    pred_geom = unary_union(pred_gdf.geometry)

    with rasterio.open(severity_path) as src:
        sev_data, sev_transform = mask(
            src, [pred_geom.__geo_interface__], crop=True, nodata=255, filled=True
        )
        sev = sev_data[0]

    lc_gdf = gpd.read_file(landcover_path)
    if lc_gdf.crs != "EPSG:32629":
        lc_gdf = lc_gdf.to_crs("EPSG:32629")
    lc_gdf = lc_gdf[lc_gdf.intersects(pred_geom)].copy()

    if lc_gdf.empty:
        print("  No land cover polygons intersect the predicted burn area")
        return None, None, None

    unique_classes = lc_gdf[landcover_class_col].unique()
    class_to_code = {cls: idx for idx, cls in enumerate(unique_classes, start=1)}
    code_to_class = {v: k for k, v in class_to_code.items()}
    lc_gdf["lc_code"] = lc_gdf[landcover_class_col].map(class_to_code)

    lc_raster = rasterize(
        zip(lc_gdf.geometry, lc_gdf["lc_code"]),
        out_shape=sev.shape,
        transform=sev_transform,
        fill=255,
        dtype=np.uint16,
    )

    valid = (sev != 255) & (lc_raster != 255)
    sev_v = sev[valid]
    lc_v = lc_raster[valid]

    if sev_v.size == 0:
        print("  No overlapping valid pixels.")
        return None, None, None

    pixel_area_ha = abs(sev_transform.a * sev_transform.e) / 10_000

    combined = sev_v.astype(np.int64) * 10000 + lc_v.astype(np.int64)
    unique_comb, counts = np.unique(combined, return_counts=True)

    rows = []
    for comb, cnt in zip(unique_comb, counts):
        sc = int(comb // 10000)
        lc = int(comb % 10000)
        rows.append({
            "severity": sc,
            "severity_label": severity_labels.get(sc, f"Unknown({sc})"),
            "landcover_label": code_to_class.get(lc, f"Unknown({lc})"),
            "pixels": int(cnt),
            "area_ha": round(cnt * pixel_area_ha, 3),
        })

    df_detail = pd.DataFrame(rows)

    df_summary = (
        df_detail.groupby(["severity", "severity_label"])
        .agg(pixels=("pixels", "sum"), area_ha=("area_ha", "sum"))
        .reset_index()
    )

    top3 = (
        df_detail
        .sort_values(["severity_label", "area_ha"], ascending=[True, False])
        .groupby("severity_label", as_index=False)
        .head(3)
    )

    print("\n=== Summary by Severity ===")
    print(df_summary.to_string(index=False))
    print("\n=== Top-3 land cover by Severity ===")
    print(top3.to_string(index=False))

    df_summary.to_csv(os.path.join(output_dir, "severity_burned_non_qa.csv"),
                       index=False)
    top3.to_csv(os.path.join(output_dir, "top3_landcover_non_qa.csv"),
                 index=False)

    return df_detail, df_summary, top3


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    # Create output dirs
    for d in [OUT_PREFIRE_DIR, OUT_POSTFIRE_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. NBR
    print("\n" + "=" * 60)
    print("STEP 1: NBR")
    print("=" * 60)
    nbr_pre = non_curated_nbr(clip_pre["B8A"], clip_pre["B12"], NBR_PRE)
    nbr_post = non_curated_nbr(clip_post["B8A"], clip_post["B12"], NBR_POST)

    # 2. dNBR
    print("\n" + "=" * 60)
    print("STEP 2: dNBR")
    print("=" * 60)
    dnbr = calculate_and_save_dnbr(NBR_PRE, NBR_POST, DNBR_PATH)

    # 3. Severity
    print("\n" + "=" * 60)
    print("STEP 3: Severity Classification")
    print("=" * 60)
    sev = classify_severity(DNBR_PATH, SEVERITY_PATH)

    # 4. Burn scar polygons
    print("\n" + "=" * 60)
    print("STEP 4: Burn Scar Polygons")
    print("=" * 60)
    burn = create_burn_scar_polygon(
        DNBR_PATH, threshold=0.1, min_area_ha=15.0,
        output_path=POLYGON_PATH,
    )

    # 5. Validation
    print("\n" + "=" * 60)
    print("STEP 5: Validation vs Reference")
    print("=" * 60)
    compare_burn_scars(POLYGON_PATH, FIRE_PATH, OUTPUT_DIR)

    # 6. Statistics
    print("\n" + "=" * 60)
    print("STEP 6: Severity Statistics")
    print("=" * 60)
    severity_area(POLYGON_PATH, SEVERITY_PATH, OUTPUT_DIR)
    severity_landcover_stats(POLYGON_PATH, SEVERITY_PATH, LANDUSE_PATH, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("NON-CURATED PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
