# A Rúa Wildfire — Burn Severity Analysis (Sentinel-2 dNBR)

**Location**: A Rúa, Valdeorras, Ourense, Galicia  
**Event**: Larouco-Seadur wildfire, August 2025 (~30,000 ha)  
**Reference**: Copernicus EMS EMSR837 (454 features, 33,003 ha)  
**Imagery**: Sentinel-2 L2A — Pre-fire 2025-08-01 (S2C) / Post-fire 2025-08-23 (S2A), Tile T29TPH

---

## Project Structure

```
├── data/
│   ├── prefire/              # Sentinel-2 bands + metadata (B04, B8A, B12, SCL, AOT, WVP, MTD_TL.xml)
│   ├── postfire/             # Same bands for post-fire scene
│   ├── terrain/              # terrain.tif (3 bands: DEM, slope, aspect)
│   ├── aoi.geojson
│   ├── true_perimeter.geojson  # Copernicus EMS reference
│   └── landuse_aoi.geojson
├── notebooks/
│   ├── get_images_with_qa.ipynb        # Data acquisition, scene selection & co-registration
│   ├── terrain.ipynb                   # DEM processing: slope, aspect, terrain stack
│   ├── non_curated_pipeline.ipynb      # Baseline: basic NBR without QA
│   ├── curated_pipeline.ipynb          # Full QA pipeline with topographic correction
│   └── report_with_severity.ipynb      # Comparative analysis & conclusions
├── output/
│   ├── non_qa/               # Non-curated results
│   └── with_qa/              # Curated results
└── README.md
```

---

## Pipeline Evolution

### Phase 1 — Non-Curated Baseline (December 2025)

- **Notebook**: [non_curated_pipeline.ipynb](./notebooks/non_curated_pipeline.ipynb)
- **Results**: [non_qa/](./output/non_qa/)
- Basic NBR calculation from raw L2A bands. No QA masks, no topographic correction.
- Threshold 0.1, min_area 15 ha.
- Showed inflated Low/Unburned classes due to noise contamination (cloud shadows, smoke, edge artifacts, non-vegetated surfaces diluting dNBR).
- Overestimated total burned area by +13.7% (37,535 ha vs 33,003 ha reference).
- Shared on LinkedIn as a baseline to highlight the importance of QA in burn severity mapping.

### Phase 2 — Curated Pipeline with QA + Topographic Correction

- **Notebook**: [curated_pipeline.ipynb](./notebooks/curated_pipeline.ipynb)
- **Results**: [with_qa/](./output/with_qa/)
- 9 QA layers applied before NBR calculation:

| Layer | Type | Purpose |
|---|---|---|
| Cloud mask | Sensor QA | Exclude clouds and cloud shadows from SCL |
| Smoke mask | Sensor QA | Detect residual smoke in post-fire imagery |
| Atmospheric validation | Diagnostic | Verify L2A correction quality (AOT/WVP) |
| Saturation mask | Radiometric QA | Exclude pixels exceeding sensor dynamic range |
| Outlier detection | Statistical QA | Flag extreme reflectance values |
| Edge mask | Geometric QA | Buffer unreliable pixels near image boundaries |
| Vegetation mask (NDVI) | Thematic QA | Restrict pre-fire analysis to vegetated areas |
| Bright surface mask | Thematic QA | Exclude bare soil, concrete, high-albedo surfaces |
| SCS+C topographic correction | Radiometric correction | Normalize B8A/B12 for terrain illumination (Soenen et al., 2005) |

- Threshold 0.3, min_area 0.1 ha, kernel (1,1).
- Topographic correction was critical: A Rúa has >900 m of elevation range (Sil valley ~300 m to Cabeza Porriñas 1,221 m).

---

## Key Results

### Detection Accuracy (vs Copernicus EMS EMSR837)

| Metric | Non-Curated | Curated |
|---|---|---|
| **IoU** | 87.3% | **89.5%** |
| **Precision** | 87.6% | **93.9%** |
| **Recall** | **99.6%** | 95.0% |
| **Area difference** | +13.7% | **+1.16%** |
| **False positives** | 4,662 ha (14.1%) | **2,005 ha (6.1%)** |
| **False negatives** | 130 ha (0.4%) | 1,646 ha (5.0%) |
| **Features** | — | 475 vs 454 ref |

### Severity Redistribution

| Class | Non-Curated | Curated | Change |
|---|---|---|---|
| High | 728 ha (2.0%) | 1,247 ha (3.9%) | **+71%** |
| Moderate-High | 14,630 ha (39.2%) | 16,117 ha (50.4%) | **+10%** |
| Moderate-Low | 15,270 ha (41.0%) | 12,791 ha (40.0%) | -16% |
| Low | 5,054 ha (13.6%) | 1,522 ha (4.8%) | -70% |
| Unburned | 1,603 ha (4.3%) | 304 ha (1.0%) | -81% |

The non-curated pipeline **underestimated fire severity**. Noise-contaminated pixels diluted dNBR values within the burn perimeter, inflating the Low and Unburned classes. After QA, the true severity distribution emerged — predominantly Moderate-High (50.4%), with more area correctly classified as High severity.

---

## Supporting Notebooks

- **[get_images_with_qa.ipynb](./notebooks/get_images_with_qa.ipynb)**: Scene acquisition from Copernicus Data Space (Sentinel Hub API), true color exploration, co-registration validation, and AOI clipping.
- **[terrain.ipynb](./notebooks/terrain.ipynb)**: DEM acquisition, slope/aspect calculation, and 3-band terrain stack generation for SCS+C correction.
- **[report_with_severity.ipynb](./notebooks/report_with_severity.ipynb)**: Full comparative analysis — area, perimeter, IoU, precision/recall, severity distribution, error analysis, and spatial visualization.

---

## Limitations

- **Perimeter gap**: Predicted 1,041 km vs reference 2,814 km. Structural limitation of Sentinel-2 at 20 m vs VHR-based Copernicus EMS cartography.
- **No water mask**: Sil river and San Martiño reservoir are in the AOI. A NDWI-based mask would reduce false positives in riparian areas.
- **Single event**: Validated against one fire only. Generalizability requires testing on additional events.
- **USGS thresholds**: dNBR severity classes not calibrated with field CBI data for Atlantic forest ecosystems.

---

## References

- Soenen, S.A., Peddle, D.R., & Coburn, C.A. (2005). SCS+C: A modified Sun-Canopy-Sensor topographic correction in forested terrain. *IEEE Transactions on Geoscience and Remote Sensing*, 43(9), 2148–2159.
- Key, C.H. & Benson, N.C. (2006). Landscape Assessment (LA). In: FIREMON: Fire Effects Monitoring and Inventory System. USDA Forest Service, RMRS-GTR-164-CD.
- Copernicus Emergency Management Service (EMS), EMSR837. European Commission.

---

See LinkedIn post for context: [Burn Severity Analysis — LinkedIn](https://www.linkedin.com/posts/augusto-salonio-442621132_wildfire-burnseverity-dnbr-activity-7421506656993210368-TSqu)