# A Rua Wildfire Burn Severity Analysis (Sentinel-2 dNBR)

## Phase 1 – Initial Results (LinkedIn post – December 2025)
- [severity_non_qa.ipynb](./notebooks/severity_non_qa.ipynb)  
- Results: [non_qa/](./output/non_qa/)  
- **Note**: This version shows inflated "High" severity (~40%) due to **border artifacts** (NIR/SWIR2 = 0 outside AOI not masked → false low NBR values).  
  This was shared on LinkedIn as a "non-curated" example to highlight QA importance.

## Corrected Version – With QA Mask
- [severity_with_qa.ipynb](./notebooks/severity_with_qa.ipynb)  
- Results: [with_qa/](./output/with_qa/)  
- **Key fix**: Valid pixel mask `(nir > 0) & (swir2 > 0)` before NBR calculation → removes edge effects.  
  → High severity significantly reduced, more realistic distribution aligned with Copernicus EMS reports.

## Comparison Notebook
- [qa_impact_comparison.ipynb](./notebooks/qa_impact_comparison.ipynb)  
  Side-by-side stats, area tables (ha per class), and visual difference maps.

See LinkedIn post for context: [https://www.linkedin.com/posts/augusto-salonio-442621132_wildfire-burnseverity-dnbr-activity-7421506656993210368-TSqu?utm_source=share&utm_medium=member_desktop&rcm=ACoAACB70zIBpz1rXQlFlhXLOTC7w4-pCdCXzw0]