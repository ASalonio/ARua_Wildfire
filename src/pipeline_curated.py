"""
Curated Wildfire Pipeline (Rigorous QA)
=======================================
Full QA pipeline for burn severity mapping.

Applies:
  - Cloud mask (SCL + buffer)
  - Smoke detection
  - Atmospheric correction validation (AOT / WVP diagnostics)
  - Saturation mask
  - Statistical outlier detection
  - Edge artifact buffer
  - Vegetation filter (NDVI on pre-fire)
  - Brightness filter (pre-fire)
  - Topographic correction (SCS+C)
  - NBR / dNBR / severity / polygon extraction / validation

Run from: c:\\Repos\\ARua_Wildfire\\
    python src/pipeline_curated.py
"""

import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import numpy.ma as ma
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes, rasterize
from scipy.ndimage import (
    binary_dilation, binary_fill_holes, binary_closing, binary_opening, label,
)
from shapely.geometry import shape
from shapely.ops import unary_union


# ═════════════════════════════════════════════════════════════
# DIRECTORIES & PATHS
# ═════════════════════════════════════════════════════════════

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "with_qa")

DATA_PREFIRE_DIR = os.path.join(DATA_DIR, "prefire")
DATA_POSTFIRE_DIR = os.path.join(DATA_DIR, "postfire")
OUT_PREFIRE_DIR = os.path.join(OUTPUT_DIR, "prefire")
OUT_POSTFIRE_DIR = os.path.join(OUTPUT_DIR, "postfire")

AOI_PATH = os.path.join(DATA_DIR, "aoi.geojson")
LANDUSE_PATH = os.path.join(DATA_DIR, "landuse_aoi.geojson")
FIRE_PATH = os.path.join(DATA_DIR, "true_perimeter.geojson")
TERRAIN_PATH = os.path.join(DATA_DIR, "terrain.tif")

META_PRE = os.path.join(DATA_PREFIRE_DIR, "MTD_TL.xml")
META_POST = os.path.join(DATA_POSTFIRE_DIR, "MTD_TL.xml")

clip_pre = {
    "B04": os.path.join(DATA_PREFIRE_DIR, "clip_red.tiff"),
    "B8A": os.path.join(DATA_PREFIRE_DIR, "clip_nir.tiff"),
    "B12": os.path.join(DATA_PREFIRE_DIR, "clip_swir2.tiff"),
    "SCL": os.path.join(DATA_PREFIRE_DIR, "clip_scl.tiff"),
    "AOT": os.path.join(DATA_PREFIRE_DIR, "clip_aot.tiff"),
    "WVP": os.path.join(DATA_PREFIRE_DIR, "clip_wvp.tiff"),
}

clip_post = {
    "B04": os.path.join(DATA_POSTFIRE_DIR, "clip_red.tiff"),
    "B8A": os.path.join(DATA_POSTFIRE_DIR, "clip_nir.tiff"),
    "B12": os.path.join(DATA_POSTFIRE_DIR, "clip_swir2.tiff"),
    "SCL": os.path.join(DATA_POSTFIRE_DIR, "clip_scl.tiff"),
    "AOT": os.path.join(DATA_POSTFIRE_DIR, "clip_aot.tiff"),
    "WVP": os.path.join(DATA_POSTFIRE_DIR, "clip_wvo.tiff"),
}

# --- Mask outputs ---
CLOUD_PRE = os.path.join(OUT_PREFIRE_DIR, "cloud_pre.tif")
CLOUD_POST = os.path.join(OUT_POSTFIRE_DIR, "cloud_post.tif")
SMOKE_PRE = os.path.join(OUT_PREFIRE_DIR, "smoke_pre.tif")
SMOKE_POST = os.path.join(OUT_POSTFIRE_DIR, "smoke_post.tif")
SAT_PRE = os.path.join(OUT_PREFIRE_DIR, "saturation_pre.tif")
SAT_POST = os.path.join(OUT_POSTFIRE_DIR, "saturation_post.tif")
OUTLIER_PRE = os.path.join(OUT_PREFIRE_DIR, "outlier_pre.tif")
OUTLIER_POST = os.path.join(OUT_POSTFIRE_DIR, "outlier_post.tif")
EDGE_PRE = os.path.join(OUT_PREFIRE_DIR, "edge_pre.tif")
EDGE_POST = os.path.join(OUT_POSTFIRE_DIR, "edge_post.tif")
VEG_PRE = os.path.join(OUT_PREFIRE_DIR, "veg_pre.tif")
BRIGHT_PRE = os.path.join(OUT_PREFIRE_DIR, "bright_pre.tif")

# --- Topographic correction outputs ---
TCOR_B8A_PRE = os.path.join(OUT_PREFIRE_DIR, "b8a_pre_tcor.tif")
TCOR_B12_PRE = os.path.join(OUT_PREFIRE_DIR, "b12_pre_tcor.tif")
TCOR_B8A_POST = os.path.join(OUT_POSTFIRE_DIR, "b8a_post_tcor.tif")
TCOR_B12_POST = os.path.join(OUT_POSTFIRE_DIR, "b12_post_tcor.tif")

# --- Products ---
NBR_PRE = os.path.join(OUT_PREFIRE_DIR, "nbr_pre_with_qa.tif")
NBR_POST = os.path.join(OUT_POSTFIRE_DIR, "nbr_post_with_qa.tif")
DNBR_PATH = os.path.join(OUTPUT_DIR, "dnbr.tif")
SEVERITY_PATH = os.path.join(OUTPUT_DIR, "severity_with_qa.tif")
POLYGON_PATH = os.path.join(OUTPUT_DIR, "burn_polygons_with_qa.gpkg")

SEVERITY_LABELS = {
    0: "Unburned",
    1: "Low",
    2: "Moderate-Low",
    3: "Moderate-High",
    4: "High",
}


# ═════════════════════════════════════════════════════════════
# QA MASKS
# ═════════════════════════════════════════════════════════════

def cloud_mask(scl_path, out_path):
    """
    Sensor QA:
    Enhanced cloud mask using SCL classes 3, 8, 9, 10 + 3px buffer.
    """
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
        meta = src.meta.copy()

    cmask = np.isin(scl, [3, 8, 9, 10]).astype(np.uint8)
    cmask = binary_dilation(cmask, iterations=3)

    cloud_fraction = np.sum(cmask) / cmask.size
    print(f"  Cloud fraction: {cloud_fraction:.1%}")

    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": -9999.0})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(cmask, 1)

    return cmask


def smoke_mask(b8a_path, b12_path, scl_path, out_path):
    """
    Sensor QA:
    Detect smoke using spectral thresholds + SCL thin cirrus cross-check.
    Critical for post-fire imagery.
    """
    with rasterio.open(b8a_path) as src:
        b8a = src.read(1).astype("float32") / 10000.0
        meta = src.meta.copy()
    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype("float32") / 10000.0
    with rasterio.open(scl_path) as src:
        scl = src.read(1).astype(np.float32)

    smoke_candidate = (
        (b8a > 0.15) & (b8a < 0.35) & (b12 < 0.2) & ((b8a - b12) > 0.02)
    )
    scl_smoke = scl == 10
    smask = smoke_candidate & scl_smoke
    smask = binary_opening(smask, iterations=1)

    smoke_fraction = np.sum(smask) / smask.size
    print(f"  Smoke fraction: {smoke_fraction:.1%}")

    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": -9999})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(smask, 1)

    return smask


def validate_atmospheric_correction(aot_path, wvp_path, b12_path):
    """
    Diagnostic sanity check for atmospheric correction (not a mask).
    Checks AOT, WVP, and negative reflectance.
    """
    issues = 0

    with rasterio.open(aot_path) as src:
        aot = src.read(1).astype("float32") / 1000.0
    if np.any(aot < 0):
        warnings.warn("Negative AOT values detected")
        issues += 1
    mean_aot = np.nanmean(aot)
    if mean_aot is not None and mean_aot > 0.4:
        warnings.warn(f"High aerosol loading (AOT = {mean_aot:.2f})")
        issues += 1

    with rasterio.open(wvp_path) as src:
        wvp = src.read(1).astype("float32") / 10000.0
    mean_wvp = np.nanmean(wvp)
    if wvp is not None and (mean_wvp > 6.0 or mean_wvp < 0):
        warnings.warn(f"Unusual water vapor values (WVP = {mean_wvp:.2f})")
        issues += 1

    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype("float32") / 10000.0
    if np.any(b12 < -0.01):
        warnings.warn("Negative reflectance detected - atmospheric overcorrection")
        issues += 1

    print(f"  Atmospheric QA issues: {issues}")
    return issues


def saturation_mask(b8a_path, b12_path, out_path, threshold=0.95):
    """Radiometric QA: flag pixels where any band exceeds threshold."""
    with rasterio.open(b8a_path) as src:
        b8a = src.read(1).astype("float32") / 10000.0
        meta = src.meta.copy()
    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype("float32") / 10000.0

    smask = ((b8a > threshold) | (b12 > threshold)).astype(np.uint8)
    print(f"  Saturation fraction: {np.mean(smask):.1%}")

    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": -9999.0})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(smask, 1)

    return smask


def detect_outliers(b8a_path, b12_path, out_path):
    """Statistics QA: detect outliers beyond 3×(p99-p1) range."""
    with rasterio.open(b8a_path) as src:
        b8a = src.read(1).astype("float32") / 10000.0
        meta = src.meta.copy()
    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype("float32") / 10000.0

    outlier_mask = np.zeros(b8a.shape, dtype=bool)

    for name, band in {"B8A": b8a, "B12": b12}.items():
        valid = band[~np.isnan(band)]
        if valid.size == 0:
            continue
        p1, p99 = np.percentile(valid, [1, 99])
        spread = p99 - p1
        lower = p1 - 3 * spread
        upper = p99 + 3 * spread
        band_outliers = (band < lower) | (band > upper)
        outlier_mask |= band_outliers
        print(f"  {name} outlier fraction: {np.sum(band_outliers) / band_outliers.size:.1%}")

    print(f"  Combined outlier fraction: {np.sum(outlier_mask) / outlier_mask.size:.2%}")

    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": -9999})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(outlier_mask.astype(np.uint8), 1)


def edge_mask(b8a_path, out_path, buffer_pixels=3):
    """Geometry QA: mask edge pixels near image boundary."""
    with rasterio.open(b8a_path) as src:
        b8a = src.read(1).astype("float32") / 10000.0
        meta = src.meta.copy()

    emask = np.zeros(b8a.shape, dtype=bool)
    emask[:buffer_pixels, :] = True
    emask[-buffer_pixels:, :] = True
    emask[:, :buffer_pixels] = True
    emask[:, -buffer_pixels:] = True

    print(f"  Edge fraction: {np.sum(emask) / emask.size:.1%}")

    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": -9999})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(emask.astype(np.uint8), 1)

    return emask


def ndvi_mask(b8a_path, b04_path, out_path):
    """Vegetation QA: pre-fire NDVI > 0.25 mask."""
    with rasterio.open(b8a_path) as src:
        b8a = src.read(1).astype("float32") / 10000.0
        meta = src.meta.copy()
    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype("float32") / 10000.0

    eps = 1e-10
    ndvi = np.zeros_like(b8a)
    ndvi_valid = (b8a + b04) > eps
    ndvi[ndvi_valid] = (b8a[ndvi_valid] - b04[ndvi_valid]) / (b8a[ndvi_valid] + b04[ndvi_valid])

    vmask = (ndvi > 0.25).astype(np.uint8)
    print(f"  Vegetated fraction: {np.sum(vmask) / vmask.size:.1%}")

    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": -9999})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(vmask, 1)

    return vmask


def bright_mask(b8a_path, b04_path, b12_path, out_path):
    """Brightness QA: exclude bright surfaces (mean reflectance >= 0.40)."""
    with rasterio.open(b8a_path) as src:
        b8a = src.read(1).astype("float32") / 10000.0
        meta = src.meta.copy()
    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype("float32") / 10000.0
    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype("float32") / 10000.0

    brightness = (b8a + b12 + b04) / 3
    bmask = (brightness < 0.40).astype(np.uint8)
    print(f"  Non-bright fraction: {np.sum(bmask) / bmask.size:.1%}")

    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": -9999})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(bmask, 1)

    return bmask


# ═════════════════════════════════════════════════════════════
# TOPOGRAPHIC CORRECTION (SCS+C)
# ═════════════════════════════════════════════════════════════

def get_solar_angles(metadata_path):
    """Extract mean solar angles from Sentinel-2 MTD_TL.xml."""
    tree = ET.parse(metadata_path)
    root = tree.getroot()
    for elem in root.iter("Mean_Sun_Angle"):
        zenith = float(elem.find("ZENITH_ANGLE").text)
        azimuth = float(elem.find("AZIMUTH_ANGLE").text)
        return zenith, azimuth
    raise ValueError(f"Mean_Sun_Angle not found in {metadata_path}")


def compute_cos_incidence(slope_deg, aspect_deg, sun_zenith_deg, sun_azimuth_deg):
    """Cosine of solar incidence angle on terrain."""
    slope = np.radians(slope_deg)
    aspect = np.radians(aspect_deg)
    sz = np.radians(sun_zenith_deg)
    sa = np.radians(sun_azimuth_deg)
    return (
        np.cos(slope) * np.cos(sz)
        + np.sin(slope) * np.sin(sz) * np.cos(sa - aspect)
    )


def align_terrain_to_band(terrain_path, band_path):
    """Reproject terrain (slope, aspect) to match a Sentinel-2 band grid."""
    with rasterio.open(band_path) as src_band:
        band_crs = src_band.crs
        band_transform = src_band.transform
        band_width = src_band.width
        band_height = src_band.height

    with rasterio.open(terrain_path) as src_t:
        slope_src = src_t.read(2)
        aspect_src = src_t.read(3)
        terrain_crs = src_t.crs
        terrain_transform = src_t.transform

    slope_aligned = np.empty((band_height, band_width), dtype=np.float32)
    aspect_aligned = np.empty((band_height, band_width), dtype=np.float32)

    reproject(source=slope_src, destination=slope_aligned,
              src_transform=terrain_transform, src_crs=terrain_crs,
              dst_transform=band_transform, dst_crs=band_crs,
              resampling=Resampling.bilinear)

    reproject(source=aspect_src, destination=aspect_aligned,
              src_transform=terrain_transform, src_crs=terrain_crs,
              dst_transform=band_transform, dst_crs=band_crs,
              resampling=Resampling.nearest)

    print(f"  Terrain aligned: {slope_src.shape} → {slope_aligned.shape}")
    return slope_aligned, aspect_aligned


def scs_c_correction(band_path, terrain_path, sun_zenith_deg, sun_azimuth_deg, out_path):
    """
    SCS+C topographic correction (Soenen et al., 2005).
    ρ_corrected = ρ_observed × (cos(θs)·cos(α) + c) / (cos(i) + c)
    """
    print(f"\n  SCS+C correction: {os.path.basename(band_path)}")

    with rasterio.open(band_path) as src:
        band = src.read(1).astype("float32")
        meta = src.meta.copy()

    slope_deg, aspect_deg = align_terrain_to_band(terrain_path, band_path)
    cos_i = compute_cos_incidence(slope_deg, aspect_deg, sun_zenith_deg, sun_azimuth_deg)

    cos_sz = np.cos(np.radians(sun_zenith_deg))
    cos_slope = np.cos(np.radians(slope_deg))
    scs_numerator = cos_sz * cos_slope

    valid = (
        (band > 0) & (cos_i > 0.1) & (slope_deg > 2.0)
        & (aspect_deg >= 0) & np.isfinite(band) & np.isfinite(cos_i)
    )
    n_valid = np.sum(valid)
    print(f"  Valid pixels for regression: {n_valid} ({n_valid/band.size:.1%})")

    if n_valid < 100:
        print("  Too few valid pixels — skipping correction")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(band, 1)
        return band

    if n_valid > 500_000:
        rng = np.random.default_rng(42)
        idx = np.where(valid.ravel())[0]
        sample_idx = rng.choice(idx, 500_000, replace=False)
        cos_i_sample = cos_i.ravel()[sample_idx]
        band_sample = band.ravel()[sample_idx]
    else:
        cos_i_sample = cos_i[valid]
        band_sample = band[valid]

    m, b = np.polyfit(cos_i_sample, band_sample, 1)
    c = b / m if abs(m) > 1e-10 else 0.0
    print(f"  Regression: m={m:.4f}, b={b:.4f}, c={c:.4f}")

    if c < 0:
        print(f"  Negative c={c:.4f} — using |c| as stabilizer")
        c = abs(c)

    corrected = np.copy(band)
    denominator = cos_i + c
    numerator = scs_numerator + c
    apply_mask = valid & (denominator > 0.05)
    corrected[apply_mask] = band[apply_mask] * (numerator[apply_mask] / denominator[apply_mask])
    corrected = np.clip(corrected, 0, 10000)

    print(f"  Corrected pixels: {np.sum(apply_mask)} ({np.sum(apply_mask)/band.size:.1%})")
    print(f"  Mean reflectance: {np.mean(band[apply_mask]):.1f} → {np.mean(corrected[apply_mask]):.1f}")

    meta.update({"dtype": "float32", "compress": "lzw"})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(corrected.astype(np.float32), 1)

    return corrected


# ═════════════════════════════════════════════════════════════
# NBR (CURATED)
# ═════════════════════════════════════════════════════════════

def curated_nbr(b8a_path, b12_path, b04_path,
                cloud_path, smoke_path, sat_path, outlier_path, edge_path,
                out_path, veg_path=None, bright_path=None):
    """
    Calculate curated NBR with 6 QA layers + optional vegetation/brightness filters.
    """
    print("\n  Calculating NBR (curated)...")

    with rasterio.open(b8a_path) as src:
        b8a = src.read(1).astype("float32") / 10000.0
        meta = src.meta.copy()
    with rasterio.open(b12_path) as src:
        b12 = src.read(1).astype("float32") / 10000.0
    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype("float32") / 10000.0

    def load_mask(path):
        with rasterio.open(path) as s:
            return s.read(1).astype(bool)

    qa_mask = (
        load_mask(cloud_path) | load_mask(smoke_path) | load_mask(sat_path)
        | load_mask(outlier_path) | load_mask(edge_path)
    )

    veg_mask = load_mask(veg_path) if veg_path is not None else np.ones(b8a.shape, dtype=bool)
    bright_m = load_mask(bright_path) if bright_path is not None else np.ones(b8a.shape, dtype=bool)

    nodata = -9999.0
    data_exists = (b8a != nodata) & (b12 != nodata) & (b04 != nodata)
    positive_reflectance = (b8a > 0) & (b12 > 0) & (b04 > 0)
    valid = data_exists & positive_reflectance

    compute_mask = valid & (~qa_mask) & veg_mask & bright_m

    nbr = np.full(b8a.shape, nodata, dtype="float32")
    nbr[compute_mask] = (
        (b8a[compute_mask] - b12[compute_mask])
        / (b8a[compute_mask] + b12[compute_mask] + 1e-10)
    )

    nbr_valid = nbr[compute_mask]
    out_of_range = np.sum((nbr_valid < -1) | (nbr_valid > 1))
    if out_of_range > 0:
        print(f"  {out_of_range} NBR values out of range — clipping")
        nbr[compute_mask] = np.clip(nbr[compute_mask], -1.0, 1.0)

    masked_count = np.sum(~compute_mask)
    print(f"  Masked pixels: {masked_count} ({masked_count / nbr.size:.1%})")

    meta.update({"driver": "GTiff", "count": 1, "dtype": "float32",
                 "nodata": nodata, "compress": "lzw"})
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

    profile.update({"dtype": "float32", "nodata": -9999.0, "compress": "lzw"})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dnbr, 1)

    print(f"  Saved to: {output_path}")
    return dnbr


# ═════════════════════════════════════════════════════════════
# SEVERITY CLASSIFICATION
# ═════════════════════════════════════════════════════════════

def classify_severity(dnbr_path, output_path):
    """USGS severity classes from dNBR (0–4, 255=nodata)."""
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

    profile.update({"dtype": "uint8", "nodata": 255, "compress": "lzw", "count": 1})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(sev, 1)

    print(f"\n  Severity saved to: {output_path}")
    return sev


# ═════════════════════════════════════════════════════════════
# BURN SCAR POLYGONS
# ═════════════════════════════════════════════════════════════

def create_burn_scar_polygon(dnbr_path, threshold, min_area_ha,
                              output_path=None, dissolve=False):
    """
    Convert dNBR raster to burn scar polygon(s).
    dissolve=True merges all into one; False keeps individual patches.
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

    burn_mask = binary_fill_holes(burn_mask).astype("uint8")
    structure = np.ones((1, 1))
    burn_mask = binary_closing(burn_mask, structure=structure).astype("uint8")

    burn_px_after = np.sum(burn_mask == 1)
    print(f"  After morphology: {burn_px_after:,} ({burn_px_after / total_valid * 100:.2f}%)")

    records = []
    for geom, value in shapes(burn_mask, mask=(burn_mask == 1), transform=transform):
        records.append({"geometry": shape(geom), "burned": int(value)})

    if not records:
        print("  No burned areas detected")
        return None

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10_000

    print(f"  Initial polygons: {len(gdf)}, total area: {gdf['area_ha'].sum():.2f} ha")

    gdf_filtered = gdf[gdf["area_ha"] >= min_area_ha].copy()
    removed = len(gdf) - len(gdf_filtered)
    if removed > 0:
        print(f"  Removed {removed} small patches (< {min_area_ha} ha)")

    if dissolve:
        merged = unary_union(gdf_filtered.geometry)
        gdf_final = gpd.GeoDataFrame(
            {
                "burned": [1],
                "area_ha": [merged.area / 10_000],
                "area_km2": [merged.area / 1_000_000],
                "perimeter_km": [merged.length / 1000],
                "n_patches": [len(gdf_filtered)],
                "dnbr_threshold": [threshold],
            },
            geometry=[merged], crs=crs,
        )
        print(f"  Final (dissolved): {gdf_final['area_ha'].iloc[0]:.2f} ha")
    else:
        gdf_filtered["area_km2"] = gdf_filtered["area_m2"] / 1_000_000
        gdf_filtered["perimeter_km"] = gdf_filtered.geometry.length / 1000
        gdf_filtered["dnbr_threshold"] = threshold
        gdf_final = gdf_filtered
        print(f"  Final: {len(gdf_final)} patches, {gdf_final['area_ha'].sum():.2f} ha total")

    if output_path:
        gdf_final.to_file(output_path, driver="GPKG")
        print(f"  Saved to: {output_path}")

    return gdf_final


# ═════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════

def compare_burn_scars(predicted_path, reference_path, output_dir):
    """Compare predicted burn scar with reference (e.g. Copernicus EMS)."""
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

    pred_compact = (4 * np.pi * pred_area) / (pred_perim ** 2)
    ref_compact = (4 * np.pi * ref_area) / (ref_perim ** 2)

    print(f"\n{'=' * 60}")
    print("BURN SCAR COMPARISON (curated)")
    print(f"{'=' * 60}")
    print(f"  Reference area:  {ref_area/10_000:>10,.2f} ha")
    print(f"  Predicted area:  {pred_area/10_000:>10,.2f} ha ({area_diff_pct:+.2f}%)")
    print(f"  IoU:             {iou:.4f} ({iou*100:.2f}%)")
    print(f"  Dice/F1:         {dice:.4f} ({dice*100:.2f}%)")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  False Pos:       {fp.area/10_000:,.2f} ha")
    print(f"  False Neg:       {fn.area/10_000:,.2f} ha")
    print(f"  Compactness ref/pred: {ref_compact:.4f} / {pred_compact:.4f}")

    os.makedirs(output_dir, exist_ok=True)

    error_gdf = gpd.GeoDataFrame(
        {"error_type": ["False Positive", "False Negative"],
         "area_ha": [fp.area / 10_000, fn.area / 10_000]},
        geometry=[fp, fn], crs=pred_gdf.crs,
    )
    error_gdf.to_file(os.path.join(output_dir, "burn_scar_errors.gpkg"), driver="GPKG")

    metrics = {
        "area_pred_ha": pred_area / 10_000, "area_ref_ha": ref_area / 10_000,
        "area_diff_pct": area_diff_pct,
        "perimeter_pred_km": pred_perim / 1000, "perimeter_ref_km": ref_perim / 1000,
        "perimeter_diff_pct": perim_diff_pct,
        "iou": iou, "dice": dice, "precision": precision, "recall": recall,
        "false_positive_ha": fp.area / 10_000, "false_negative_ha": fn.area / 10_000,
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "comparison_metrics.csv"), index=False
    )
    return metrics


# ═════════════════════════════════════════════════════════════
# SEVERITY STATISTICS
# ═════════════════════════════════════════════════════════════

def severity_area(predicted_polygon_path, severity_path, severity_labels=None):
    """Area by severity class inside predicted burn perimeter."""
    if severity_labels is None:
        severity_labels = SEVERITY_LABELS

    pred_gdf = gpd.read_file(predicted_polygon_path)
    if pred_gdf.crs != "EPSG:32629":
        pred_gdf = pred_gdf.to_crs("EPSG:32629")

    pred_geom = unary_union(pred_gdf.geometry)
    shapes_geom = (
        [pred_geom.__geo_interface__] if pred_geom.geom_type != "GeometryCollection"
        else [g.__geo_interface__ for g in pred_geom.geoms]
    )

    with rasterio.open(severity_path) as src:
        sev_data, sev_transform = mask(src, shapes_geom, crop=True,
                                        nodata=255, filled=False)
        sev = sev_data[0]
        sev_meta = src.meta.copy()
        sev_meta.update({"height": sev.shape[0], "width": sev.shape[1],
                         "transform": sev_transform})

    sev_flat = sev[~sev.mask]
    if sev_flat.size == 0:
        print("  No valid pixels inside predicted area.")
        return None

    pixel_area_ha = abs(sev_meta["transform"].a * sev_meta["transform"].e) / 10_000
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
    print(f"\n  Severity area breakdown:")
    print(df.to_string(index=False))
    return df


def severity_landcover_stats(predicted_polygon_path, severity_path,
                              landcover_path, severity_labels=None,
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
        out_shape=sev.shape, transform=sev_transform,
        fill=255, dtype=np.uint16, all_touched=False,
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
        df_detail.sort_values(["severity_label", "area_ha"], ascending=[True, False])
        .groupby("severity_label", as_index=False).head(3)
    )

    print("\n=== Summary by Severity ===")
    print(df_summary.to_string(index=False))
    print("\n=== Top-3 land cover by Severity ===")
    print(top3.to_string(index=False))

    return df_detail, df_summary, top3


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    # Create output dirs
    for d in [OUT_PREFIRE_DIR, OUT_POSTFIRE_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── STEP 1: QA Masks ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: QA Masks")
    print("=" * 60)

    print("\n[Pre-fire]")
    cloud_mask(clip_pre["SCL"], CLOUD_PRE)
    smoke_mask(clip_pre["B8A"], clip_pre["B12"], clip_pre["SCL"], SMOKE_PRE)
    validate_atmospheric_correction(clip_pre["AOT"], clip_pre["WVP"], clip_pre["B12"])
    saturation_mask(clip_pre["B8A"], clip_pre["B12"], SAT_PRE, threshold=0.95)
    detect_outliers(clip_pre["B8A"], clip_pre["B12"], OUTLIER_PRE)
    edge_mask(clip_pre["B8A"], EDGE_PRE, 3)
    ndvi_mask(clip_pre["B8A"], clip_pre["B04"], VEG_PRE)
    bright_mask(clip_pre["B8A"], clip_pre["B04"], clip_pre["B12"], BRIGHT_PRE)

    print("\n[Post-fire]")
    cloud_mask(clip_post["SCL"], CLOUD_POST)
    smoke_mask(clip_post["B8A"], clip_post["B12"], clip_post["SCL"], SMOKE_POST)
    validate_atmospheric_correction(clip_post["AOT"], clip_post["WVP"], clip_post["B12"])
    saturation_mask(clip_post["B8A"], clip_post["B12"], SAT_POST, threshold=0.95)
    detect_outliers(clip_post["B8A"], clip_post["B12"], OUTLIER_POST)
    edge_mask(clip_post["B8A"], EDGE_POST, 3)

    # ── STEP 2: Topographic Correction ────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Topographic Correction (SCS+C)")
    print("=" * 60)

    sun_z_pre, sun_a_pre = get_solar_angles(META_PRE)
    sun_z_post, sun_a_post = get_solar_angles(META_POST)
    print(f"  Pre-fire  solar: zenith={sun_z_pre:.2f}°, azimuth={sun_a_pre:.2f}°")
    print(f"  Post-fire solar: zenith={sun_z_post:.2f}°, azimuth={sun_a_post:.2f}°")

    scs_c_correction(clip_pre["B8A"], TERRAIN_PATH, sun_z_pre, sun_a_pre, TCOR_B8A_PRE)
    scs_c_correction(clip_pre["B12"], TERRAIN_PATH, sun_z_pre, sun_a_pre, TCOR_B12_PRE)
    scs_c_correction(clip_post["B8A"], TERRAIN_PATH, sun_z_post, sun_a_post, TCOR_B8A_POST)
    scs_c_correction(clip_post["B12"], TERRAIN_PATH, sun_z_post, sun_a_post, TCOR_B12_POST)

    # ── STEP 3: NBR ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: NBR (curated)")
    print("=" * 60)

    curated_nbr(
        TCOR_B8A_PRE, TCOR_B12_PRE, clip_pre["B04"],
        CLOUD_PRE, SMOKE_PRE, SAT_PRE, OUTLIER_PRE, EDGE_PRE,
        NBR_PRE, VEG_PRE, BRIGHT_PRE,
    )
    curated_nbr(
        TCOR_B8A_POST, TCOR_B12_POST, clip_post["B04"],
        CLOUD_POST, SMOKE_POST, SAT_POST, OUTLIER_POST, EDGE_POST,
        NBR_POST,
        veg_path=None, bright_path=None,
    )

    # ── STEP 4: dNBR ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: dNBR")
    print("=" * 60)
    calculate_and_save_dnbr(NBR_PRE, NBR_POST, DNBR_PATH)

    # ── STEP 5: Severity ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Severity Classification")
    print("=" * 60)
    classify_severity(DNBR_PATH, SEVERITY_PATH)

    # ── STEP 6: Burn Scar Polygons ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Burn Scar Polygons")
    print("=" * 60)
    create_burn_scar_polygon(
        DNBR_PATH, threshold=0.3, min_area_ha=0.1,
        output_path=POLYGON_PATH, dissolve=False,
    )

    # ── STEP 7: Validation ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Validation vs Reference")
    print("=" * 60)
    compare_burn_scars(POLYGON_PATH, FIRE_PATH, OUTPUT_DIR)

    # ── STEP 8: Severity Statistics ──────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Severity Statistics")
    print("=" * 60)
    severity_area(POLYGON_PATH, SEVERITY_PATH)
    severity_landcover_stats(POLYGON_PATH, SEVERITY_PATH, LANDUSE_PATH)

    print("\n" + "=" * 60)
    print("CURATED PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
