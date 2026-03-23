"""
Test local functions from download.py using existing data.
Run from: c:\\Repos\\ARua_Wildfire\\
    python src/test_local.py
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import download_v2 as download

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data")
AOI  = os.path.join(DATA, "aoi.geojson")

# --------------------------------------------------
# TEST 1: clip()
# --------------------------------------------------
print("\n" + "="*60)
print("TEST 1: clip()")
print("="*60)

jp2_pre  = os.path.join(DATA, "prefire",  "T29TPH_20250801T112131_B8A_20m.jp2")
jp2_post = os.path.join(DATA, "postfire", "T29TPH_20250823T112131_B8A_20m.jp2")

# SCL bands for stable-pixel filtering
scl_pre  = os.path.join(DATA, "prefire",  "clip_scl.tiff")
scl_post = os.path.join(DATA, "postfire", "clip_scl.tiff")

with tempfile.TemporaryDirectory() as tmp:
    out_pre  = os.path.join(tmp, "pre_nir.tiff")
    out_post = os.path.join(tmp, "post_nir.tiff")

    arr_pre,  _ = download.clip(jp2_pre,  out_pre,  AOI)
    arr_post, _ = download.clip(jp2_post, out_post, AOI)

    print(f"  pre  shape: {arr_pre.shape}")
    print(f"  post shape: {arr_post.shape}")

    # --------------------------------------------------
    # TEST 2: validate_coregistration()
    # --------------------------------------------------
    print("\n" + "="*60)
    print("TEST 2: validate_coregistration()")
    print("="*60)

    # With SCL filtering (if clip_scl.tiff exist)
    if os.path.exists(scl_pre) and os.path.exists(scl_post):
        aligned = download.validate_coregistration(
            out_pre, out_post,
            scl_pre_path=scl_pre,
            scl_post_path=scl_post
        )
    else:
        print("  SCL clips not found, running without SCL filtering")
        aligned = download.validate_coregistration(out_pre, out_post)

    print("\nRESULT:", "ALIGNED ✓" if aligned else "MISALIGNED ✗")