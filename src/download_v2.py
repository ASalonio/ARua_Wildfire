import os
import time
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape
from scipy.stats import pearsonr
from scipy.signal import correlate2d
from tqdm import tqdm

# -----------------------------
# PRE-FIRE and POST-FIRE QUERY in Catalog API
# primary objective: get id that completely cover AOI
# secondary objective: get id that partially cover AOI, and then mosaic
# -----------------------------

def query_catalog(params, token, oauth, aoi_polygon, period="pre"):
    """
    Consulta el Catalog API de Copernicus y clasifica las escenas .SAFE
    en dos listas según su cobertura sobre la AOI.

    Returns:
        tuple: (full_coverage, partial_coverage)
            - full_coverage: escenas que cubren el 100% de la AOI,
              ordenadas por fecha desc y luego por menor nubosidad.
            - partial_coverage: escenas que cubren parcialmente la AOI,
              ordenadas por mayor % de cobertura y luego por menor nubosidad.
    """
    catalog_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    headers = {
        "Authorization": f"Bearer {token['access_token']}",
        "Content-Type": "application/json",
    }

    prefix = period  # "pre" or "post"
    start = params[f"{prefix}_fire_start_date"]
    end = params[f"{prefix}_fire_end_date"]
    cloud_max = params[f"{prefix}_fire_cloud"]

    json_search = {
        "bbox": params["bbox"],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "collections": [params["collection_id"]],
        "limit": 50,
        "filter": f"eo:cloud_cover <= {cloud_max}",
        "fields": {
            "include": [
                "id",
                "properties.datetime",
                "properties.eo:cloud_cover",
            ]
        },
    }

    response = oauth.post(catalog_url, headers=headers, json=json_search)
    response.raise_for_status()

    features = response.json().get("features", [])
    full_coverage = []
    partial_coverage = []

    for f in features:
        tile_polygon = shape(f["geometry"])
        cloud = f["properties"].get("eo:cloud_cover", None)
        dt = f["properties"]["datetime"]

        entry = {
            "id": f["id"],
            "datetime": dt,
            "date": dt[:10],
            "cloud_cover": cloud,
            "geometry": f["geometry"],
        }

        if aoi_polygon.within(tile_polygon):
            entry["coverage_pct"] = 100.0
            full_coverage.append(entry)
        else:
            intersection = aoi_polygon.intersection(tile_polygon)
            if intersection.is_empty:
                continue  # sin solapamiento, descartamos
            coverage_pct = 100.0 * intersection.area / aoi_polygon.area
            entry["coverage_pct"] = round(coverage_pct, 2)
            partial_coverage.append(entry)

    # Full: más recientes primero, empate por menor nubosidad
    full_coverage.sort(key=lambda e: (-pd.Timestamp(e["datetime"]).timestamp(),
                                       e["cloud_cover"] or 0))
    # Partial: mayor cobertura primero, empate por menor nubosidad
    partial_coverage.sort(key=lambda e: (-e["coverage_pct"],
                                          e["cloud_cover"] or 0))

    return full_coverage, partial_coverage

# -----------------------------
# Get PRE-FIRE and POST-FIRE Scenes in OA API
# -----------------------------

def get_product_uuids(scenes, token):
    """
    Consulta OData API de Copernicus para obtener los UUIDs
    de cada escena .SAFE.

    Args:
        scenes: lista de dicts con clave 'id' (nombre .SAFE)
                (output de query_catalog, ya sea full o partial)
        token: access token string

    Returns:
        dict: {safe_name: uuid}
    """
    uuids = {}
    headers = {"Authorization": f"Bearer {token}"}
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    for scene in scenes:
        safe_name = scene["id"]
        url = f"{base_url}?$filter=Name eq '{safe_name}'&$select=Id,Name"

        try:
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code == 200:
                data = resp.json().get("value", [])
                if data:
                    uuids[safe_name] = data[0]["Id"]
                    print(f"UUID for {safe_name}: {uuids[safe_name]}")
                else:
                    print(f"No product found for {safe_name}")
            elif resp.status_code == 401:
                print("Token expired or invalid → need to refresh")
                break
            else:
                print(f"Error {resp.status_code} for {safe_name}: {resp.text[:200]}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed for {safe_name}: {e}")

        time.sleep(1)

    return uuids

# -----------------------------
# DOWNLOAD
# -----------------------------

def download_products(uuids, token, output_dir="downloads"):
    """
    Descarga productos .SAFE desde OData API de Copernicus.

    Args:
        uuids: dict {safe_name: uuid} (output de get_product_uuids)
        token: access token string
        output_dir: directorio destino para los .zip

    Returns:
        dict: {safe_name: filepath} de descargas completadas
    """
    os.makedirs(output_dir, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    downloaded = {}

    for safe_name, uuid in uuids.items():
        url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({uuid})/$value"
        print(f"\nStarting download for: {safe_name}")

        session = requests.Session()
        session.headers.update(headers)

        try:
            response = session.get(url, stream=True, timeout=60)

            if response.status_code == 200:
                content_disp = response.headers.get("Content-Disposition", "")
                filename = f"{safe_name}.zip"
                if "filename=" in content_disp:
                    filename = content_disp.split("filename=")[-1].strip('"')

                filepath = os.path.join(output_dir, filename)
                total_size = int(response.headers.get("Content-Length", 0))
                print(f"Downloading → {filename} ({total_size / 1e9:.2f} GB)")

                with open(filepath, "wb") as file:
                    if total_size > 0:
                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=safe_name[:30],
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    file.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)

                downloaded[safe_name] = filepath
                print(f"Download complete: {filepath}")

            elif response.status_code == 401:
                print("→ 401 Unauthorized: Token expired. Refresh and retry.")
                break
            else:
                print(f"Failed → Status: {response.status_code}")
                print(response.text[:400])

        except requests.exceptions.RequestException as e:
            print(f"Network error for {safe_name}: {e}")
        except Exception as e:
            print(f"Unexpected error for {safe_name}: {e}")
        finally:
            session.close()

        time.sleep(5)

    print(f"\nDownloads completed: {len(downloaded)}/{len(uuids)}")
    return downloaded

# -----------------------------
# UNZIP + EXTRACT BANDS
# -----------------------------

def extract_bands(zip_path, output_dir, resolution="20m"):
    """
    Descomprime un .SAFE.zip y localiza las bandas a la resolución deseada.
    
    Returns:
        dict: {"B02": path, "B03": path, ..., "SCL": path}
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    bands = {}
    target_bands = ["B02", "B03", "B04", "B8A", "B12", "SCL", "AOT", "WVP"]

    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".jp2") and resolution in f:
                for band in target_bands:
                    if f"_{band}_" in f and band not in bands:
                        bands[band] = os.path.join(root, f)

    return bands

# -----------------------------
# CLIP at AOI
# -----------------------------

def clip(scene_path, out_path, geometries, nodata=-9999.0): #use a numeric nodata, not NaN

    # Load AOI GeoJSON
    aoi_gdf = gpd.read_file(geometries)
    if aoi_gdf.crs != "EPSG:32629":
        aoi_gdf = aoi_gdf.to_crs("EPSG:32629")

    geom = [aoi_gdf.geometry.iloc[0]]  # assuming single polygon

    with rasterio.open(scene_path) as src:
        # Use rasterio.mask directly (unless mask_with_geom does something special)
        src_nodata = src.nodata
        if src_nodata is None:
            src_nodata = 0 #Sentinel-2 JP2 usually uses 0
            
        clipped, transform = mask(
            src,
            geom,
            crop=True,
            filled=True, #Returns a MaskedArray
            nodata=src_nodata
        )

    # Convert AFTER masking
    clipped = clipped.astype("float32")

    # Remap nodata AFTER masking
    clipped[clipped == src_nodata] = nodata

    meta = src.meta.copy()
    meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "transform": transform,
        "nodata": nodata
    })

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)

    return clipped, transform

# -----------------------------
# CO-REGISTRATION VALIDAYION
# -----------------------------

def validate_coregistration(pre_fire_path, post_fire_path,
                             scl_pre_path=None, scl_post_path=None,
                             sample_n=1000, nodata=-9999.0):
    """
    Verify that pre and post-fire images are properly aligned.

    v2 improvements:
      1. Compares rasterio transforms directly (geometric check).
      2. Uses SCL masks to sample only stable land-cover classes
         (vegetation, bare soil, water) — excludes burned, cloud, shadow.
      3. Cross-correlation window avoids burned areas.
      4. Normalizes windows to reduce brightness-change effects.
      5. Works on both full tiles and clipped images.

    Args:
        pre_fire_path:  path to pre-fire B8A raster (.jp2 or .tiff)
        post_fire_path: path to post-fire B8A raster (.jp2 or .tiff)
        scl_pre_path:   (optional) path to pre-fire SCL raster
        scl_post_path:  (optional) path to post-fire SCL raster
        sample_n:       number of random pixels for correlation
        nodata:         nodata value used in clipped tiffs

    Returns:
        bool: True if images are well-aligned
    """

    print("\n" + "=" * 70)
    print("CO-REGISTRATION VALIDATION (v2)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Read B8A bands + metadata
    # ------------------------------------------------------------------
    with rasterio.open(pre_fire_path) as src_pre:
        b8a_pre = src_pre.read(1).astype("float32")
        transform_pre = src_pre.transform
        crs_pre = src_pre.crs

    with rasterio.open(post_fire_path) as src_post:
        b8a_post = src_post.read(1).astype("float32")
        transform_post = src_post.transform
        crs_post = src_post.crs

    # ------------------------------------------------------------------
    # 2. Geometric check: transforms + CRS
    # ------------------------------------------------------------------
    print("\n--- Geometric Check ---")
    print(f"  CRS pre : {crs_pre}")
    print(f"  CRS post: {crs_post}")

    if crs_pre != crs_post:
        print("  WARNING: CRS mismatch!")

    dx = abs(transform_pre.c - transform_post.c)
    dy = abs(transform_pre.f - transform_post.f)
    res_x = abs(transform_pre.a)

    origin_offset_px = np.sqrt(dx**2 + dy**2) / res_x

    print(f"  Pixel size: {res_x}m")
    print(f"  Origin offset: {origin_offset_px:.4f} pixels")

    if origin_offset_px < 0.01:
        print("  → Origins match perfectly (same grid)")
    elif origin_offset_px < 1.0:
        print("  → Sub-pixel origin offset (acceptable)")
    else:
        print(f"  → WARNING: Origin offset {origin_offset_px:.2f} px")

    if b8a_pre.shape != b8a_post.shape:
        print(f"  ERROR: Dimensions don't match: {b8a_pre.shape} vs {b8a_post.shape}")
        print("=" * 70)
        return False

    print(f"  Dimensions: {b8a_pre.shape} ✓")

    # ------------------------------------------------------------------
    # 3. Build stable-pixel mask
    # ------------------------------------------------------------------
    # SCL classes considered stable:
    #   4 = vegetation, 5 = bare soil, 6 = water
    STABLE_SCL = {4, 5, 6}

    valid_mask = (b8a_pre > 0) & (b8a_post > 0)
    if nodata is not None:
        valid_mask &= (b8a_pre != nodata) & (b8a_post != nodata)

    scl_used = False

    if scl_pre_path and scl_post_path:
        try:
            with rasterio.open(scl_pre_path) as src:
                scl_pre = src.read(1)
            with rasterio.open(scl_post_path) as src:
                scl_post = src.read(1)

            stable_pre  = np.isin(scl_pre, list(STABLE_SCL))
            stable_post = np.isin(scl_post, list(STABLE_SCL))
            stable_mask = valid_mask & stable_pre & stable_post

            n_stable = np.count_nonzero(stable_mask)
            n_valid  = np.count_nonzero(valid_mask)
            print(f"\n--- SCL Filtering ---")
            print(f"  Valid pixels:  {n_valid:,}")
            print(f"  Stable pixels: {n_stable:,} ({100*n_stable/max(n_valid,1):.1f}%)")

            if n_stable >= sample_n:
                valid_mask = stable_mask
                scl_used = True
            else:
                print(f"  → Not enough stable pixels ({n_stable} < {sample_n}), using all valid pixels")

        except Exception as e:
            print(f"  WARNING: Could not read SCL files ({e}), using all valid pixels")

    if not scl_used:
        print(f"\n  Using all valid pixels: {np.count_nonzero(valid_mask):,}")

    # ------------------------------------------------------------------
    # 4. Pearson correlation on stable pixels
    # ------------------------------------------------------------------
    valid_indices = np.where(valid_mask)

    if len(valid_indices[0]) < 10:
        print("  ERROR: Not enough valid pixels for correlation")
        print("=" * 70)
        return False

    actual_n = min(sample_n, len(valid_indices[0]))
    sample_idx = np.random.choice(len(valid_indices[0]), actual_n, replace=False)
    sample_rows = valid_indices[0][sample_idx]
    sample_cols = valid_indices[1][sample_idx]

    pre_sample  = b8a_pre[sample_rows, sample_cols]
    post_sample = b8a_post[sample_rows, sample_cols]

    correlation, p_value = pearsonr(pre_sample, post_sample)

    label = "stable" if scl_used else "valid"
    print(f"\n--- Correlation Analysis ({actual_n} {label} pixels) ---")
    print(f"  Pearson r: {correlation:.4f}")
    print(f"  P-value:   {p_value:.2e}")

    status = True

    if correlation > 0.85:
        print("  → Strong correlation — well-aligned ✓")
    elif correlation > 0.7:
        print("  → Moderate correlation — acceptable")
    else:
        print("  → Weak correlation — possible co-registration problem")
        if not scl_used:
            print("    (Tip: provide SCL paths to exclude burned pixels)")
        status = False

    # ------------------------------------------------------------------
    # 5. Cross-correlation offset
    # ------------------------------------------------------------------
    print(f"\n--- Spatial Offset Check ---")

    window_size = 256

    if scl_used:
        # Find the 256×256 window with the most stable pixels
        best_count = 0
        best_r, best_c = b8a_pre.shape[0] // 2, b8a_pre.shape[1] // 2

        rows_max = b8a_pre.shape[0] - window_size
        cols_max = b8a_pre.shape[1] - window_size

        if rows_max > 0 and cols_max > 0:
            for r_start in np.linspace(0, rows_max, 5, dtype=int):
                for c_start in np.linspace(0, cols_max, 5, dtype=int):
                    count = np.count_nonzero(
                        valid_mask[r_start:r_start+window_size,
                                   c_start:c_start+window_size]
                    )
                    if count > best_count:
                        best_count = count
                        best_r, best_c = r_start, c_start

            print(f"  Best window at ({best_r}, {best_c}) with {best_count} stable pixels")
        else:
            print(f"  Image too small for {window_size}x{window_size} window")
            best_r, best_c = 0, 0
            window_size = min(b8a_pre.shape[0], b8a_pre.shape[1], window_size)
    else:
        best_r = min(b8a_pre.shape[0] // 2, b8a_pre.shape[0] - window_size)
        best_c = min(b8a_pre.shape[1] // 2, b8a_pre.shape[1] - window_size)
        best_r = max(best_r, 0)
        best_c = max(best_c, 0)

    window_pre = b8a_pre[best_r:best_r+window_size,
                          best_c:best_c+window_size]
    window_post = b8a_post[best_r:best_r+window_size,
                            best_c:best_c+window_size]

    # Normalize to reduce brightness-change effects
    def _normalize(arr):
        a = arr.copy()
        a[a <= 0] = np.nan
        mean, std = np.nanmean(a), np.nanstd(a)
        if std > 0:
            a = (a - mean) / std
        return np.nan_to_num(a, nan=0.0)

    xcorr = correlate2d(_normalize(window_pre), _normalize(window_post), mode='same')
    max_loc = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    center = (window_size // 2, window_size // 2)

    offset_row = max_loc[0] - center[0]
    offset_col = max_loc[1] - center[1]
    offset_pixels = np.sqrt(offset_row**2 + offset_col**2)

    print(f"  Row offset: {offset_row} pixels")
    print(f"  Col offset: {offset_col} pixels")
    print(f"  Total offset: {offset_pixels:.2f} pixels")

    if offset_pixels < 0.5:
        print("  → Excellent alignment (<0.5 pixel) ✓")
    elif offset_pixels < 1.0:
        print("  → Good alignment (<1 pixel) ✓")
    elif offset_pixels < 1.5:
        print("  → Acceptable for dNBR (<1.5 pixel, within ESA spec)")
    else:
        print(f"  → Offset {offset_pixels:.1f} pixels — may affect dNBR")
        status = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"  RESULT: {'ALIGNED ✓' if status else 'MISALIGNED ✗'}")
    if status and not scl_used and correlation < 0.8:
        print("  NOTE: Correlation may be low due to fire-induced changes.")
        print("        Provide SCL paths for a more accurate assessment.")
    print("=" * 70)

    return status