import os
import geopandas as gpd
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from secrets import CLIENT_ID, CLIENT_SECRET
import download

# -----------------------
# DIRECTORIES
# -----------------------

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

AOI_PATH = os.path.join(DATA_DIR, "aoi.geojson")

PERIODS = {
    "pre": {
        "data_dir": os.path.join(DATA_DIR, "prefire"),
        "out_dir": os.path.join(OUTPUT_DIR, "prefire"),
    },
    "post": {
        "data_dir": os.path.join(DATA_DIR, "postfire"),
        "out_dir": os.path.join(OUTPUT_DIR, "postfire"),
    },
}

for p in PERIODS.values():
    os.makedirs(p["data_dir"], exist_ok=True)
    os.makedirs(p["out_dir"], exist_ok=True)

# -----------------------
# BAND CONFIG
# -----------------------

TARGET_BANDS = ["B02", "B03", "B04", "B8A", "B12", "SCL", "AOT", "WVP"]

BAND_ALIASES = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B8A": "nir",
    "B12": "swir2",
    "SCL": "scl",
    "AOT": "aot",
    "WVP": "wvp",
}

# -----------------------
# BOUNDING BOX
# -----------------------

aoi_gdf = gpd.read_file(AOI_PATH)

if aoi_gdf.crs.to_epsg() != 4326:
    aoi_gdf = aoi_gdf.to_crs(4326)

# Assuming the GeoJSON has a single polygon
aoi_polygon = aoi_gdf.geometry.iloc[0]

# Bounding box (bbox)
minx, miny, maxx, maxy = aoi_gdf.total_bounds

# -----------------------
# AUTHENTICATION
# -----------------------

client = BackendApplicationClient(client_id=CLIENT_ID)
oauth = OAuth2Session(client=client)

token = oauth.fetch_token(
    token_url="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    client_secret=CLIENT_SECRET,
    include_client_id=True,
)

params = {
    "bbox" : [minx, miny, maxx, maxy],
    "collection_id" : "sentinel-2-12a",
    "pre_fire_start_date" : "2025-07-15",
    "pre_fire_end_date" : "2025-08-15",
    "pre_fire_cloud" : 15,
    "post_fire_start_date" : "2025-08-18",
    "post_fire_end_date" : "2025-09-15",
    "post_fire_cloud" : 20
}

# --- PIPELINE EXECUTION ---

for period in ["pre", "post"]:
    dirs = PERIODS[period]

    # 1. Query catalog
    full, partial = download.query_catalog(params, token, oauth, aoi_polygon, period=period)

    # 2. Get UUIDs (priorizamos full coverage, tomamos la más reciente)
    selected = full[:1] if full else partial  # ajustar lógica de mosaico si es partial
    uuids = download.get_product_uuids(selected, token["access_token"])

    # 3. Download
    downloaded = download.download_products(uuids, token["access_token"], output_dir=dirs["data_dir"])

    # 4. Extract bands from .SAFE zip
    for safe_name, zip_path in downloaded.items():
        bands = download.extract_bands(zip_path, dirs["data_dir"], resolution="20m")
        # bands = {"B02": "/path/to.jp2", "B03": ..., }
    
    #Si download tiene mas de un .SAFE (caso partial/mosaico), el for sobreescribe bands en cada iteración y solo te quedas con las bandas del último zip.
    #Para el caso actual (full coverage, full[:1] no es problema, pero conviene tenerlo en cuenta para cuando implementes el mosaico)

    # 5. Clip all bands with AOI
    clipped = {}
    for band_name, jp2_path in bands.items():
        alias = BAND_ALIASES.get(band_name, band_name.lower())
        clip_path = os.path.join(dirs["data_dir"], f"clip_{alias}.tiff")
        download.clip(jp2_path, clip_path, AOI_PATH)
        clipped[band_name] = clip_path

    # Store for later use
    PERIODS[period]["bands"] = bands
    PERIODS[period]["clipped"] = clipped

# 6. Validate co-registration (using clipped NIR)
is_aligned = download.validate_coregistration(
    PERIODS["pre"]["clipped"]["B8A"],
    PERIODS["post"]["clipped"]["B8A"],
)

if is_aligned:
    print("Co-registration OK → proceeding to dNBR")
else:
    print("Co-registration FAILED → review images")