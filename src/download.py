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

def validate_coregistration(pre_fire_path, post_fire_path, sample_n=1000):
    """ 
    Verify that pre and post-fire images are properly aligned
    """
    
    print("\n" + "="*70)
    print("CO-REGISTRATION VALIDATION")
    print("="*70)

    with rasterio.open(pre_fire_path) as src_pre:
        b8a_pre = src_pre.read(1).astype("float32")

    with rasterio.open(post_fire_path) as src_post:
        b8a_post = src_post.read(1).astype("float32")
    
    # Check if dimensions match
    if b8a_pre.shape != b8a_post.shape:
        print("  ERROR: Image dimensions don't match")
        print(f" Pre-fire: {b8a_pre.shape}")
        print(f" Post-fire: {b8a_post.shape}")

    print(f" Dimensions match: {b8a_pre.shape}")

    # Sampel random pixels
    valid_mask = (b8a_pre > 0) & (b8a_post > 0)
    valid_indices = np.where(valid_mask)

    if len(valid_indices[0]) < sample_n:
        sample_n = len(valid_indices[0])

    sample_idx = np.random.choice(len(valid_indices[0]), sample_n, replace=False)
    sample_rows = valid_indices[0][sample_idx]
    sample_cols = valid_indices[1][sample_idx]

    pre_sample = b8a_pre[sample_rows, sample_cols]
    post_sample = b8a_post[sample_rows, sample_cols]

    #Calculate correlation (stable features should correlate)

    correlation, p_value = pearsonr(pre_sample, post_sample)

    print(f" \n Correlation Analysis: ({sample_n} sample pixels):")
    print(f"    Pearson correlation: {correlation:.4f}")
    print(f"    P-value: {p_value: .2e} ")

    if correlation > 0.85:
        print(f" Strong correlation - images well-aligned")
        status = True
    
    elif correlation > 0.7:
        print(f" Moderate correlation - check for issues")
        status= True

    else:
        print(f" Weak correlation - co-registration problem!")
        status= False

    #Check spatial offset (cross-correlation)
    #Sample a small window for efficiency

    window_size = 256
    center_row = b8a_pre.shape[0]  // 2
    center_col = b8a_pre.shape[1] // 2

    window_pre = b8a_pre[
        center_row:center_row+window_size,
        center_col:center_col+window_size
    ]

    window_post = b8a_post[
        center_row:center_row+window_size,
        center_col:center_col+window_size
    ]

    # Simple offset check (should be near zero)
    xcorr = correlate2d(window_pre, window_post, mode='same')
    max_loc = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    center = (window_size // 2, window_size // 2)

    offset_row = max_loc[0] - center[0]
    offset_col = max_loc[1] - center[1]
    offset_pixels = np.sqrt(offset_row**2 + offset_col**2)

    print(f" \n Spatial Offset Check:")
    print(f"    Row offset: {offset_row} pixels")
    print(f"    Col offset: {offset_col} pixels")
    print(f"    Total offset: {offset_pixels:.2f} pixels")

    if offset_pixels < 0.5:
        print(f" Excellent alignment (<0.5 pixel)")

    elif offset_pixels < 1.0:
        print(f" Good alignment (<1 pixel)")

    else:
        print(f" Offset >{offset_pixels:.1f} pixels - may affect dNBR")
        status = False

    print("\n" + "="*70)
    
    return status