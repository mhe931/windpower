# Licences and attribution

Nothing in this repository redistributes third-party data. Real-data outputs (not produced so far) would
derive from the sources below and must carry the attributions; synthetic outputs contain no third-party data.

| Component | Licence | Attribution to show | Verified? |
|---|---|---|---|
| ERA5 (Copernicus Climate Change Service) | Copernicus licence, CC BY 4.0 compatible | "Contains modified Copernicus Climate Change Service information" | assumed |
| OpenStreetMap data | ODbL 1.0 | "© OpenStreetMap contributors" | assumed |
| Finnish wind turbine layer (ArcGIS) | per layer metadata (`copyrightText`, `licenseInfo`) captured at fetch time | as stated by the layer | to be verified - not captured yet |
| Fingrid open data | CC BY 4.0 | "Fingrid Oyj open data" | to be verified at https://data.fingrid.fi |
| atlite | MIT | - | verified (installed package metadata) |
| CM SAF SARAH-3 ICDR | CM SAF / EUMETSAT terms as in the file header (`license` attribute) | "Pfeifroth et al. (2023), SARAH-3, DOI 10.5676/EUM_SAF_CM/SARAH/V003" | text copied from the files, not interpreted |
| Fingrid datasets 267, 248, 362 | CC BY 4.0 | "Fingrid Oyj open data" | verified on the public dataset pages by the lead (2026-11-27) |
| Statistics Finland population grid + municipalities | CC BY 4.0 (from the solar report) | "Statistics Finland" | **not stated by the WFS service or its metadata record** - unverified |
| Suomen Uusiutuvat ry plant list | none stated in the workbook | "Suomen Uusiutuvat ry (2025)" | ask the owner before redistributing; only the 30-plant aggregate is used |
| This repository's code | no licence file yet | - | decision pending |

The licence block in `manifest.json` (and in `site/v1/manifest.json`) repeats this table together with a status for
each entry. Derivative works (the website artifact) must keep the attribution of every real source they use.
