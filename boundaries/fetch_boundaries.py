"""Fetch land and EEZ polygons for Denmark, Belgium and Finland, in the same format as the Germany files
(Germany: germany/germany_land.geojson, germany/germany_eez.geojson): land = Natural Earth 10m admin-0 countries, EEZ = Marine Regions (VLIZ) EEZ v11 via WFS.
Writes <country>/<country>_land.geojson and <country>/<country>_eez.geojson next to this script. Needs network access."""
import io
import os
import zipfile

import geopandas as gpd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
NE_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
WFS = "https://geo.vliz.be/geoserver/wfs"
COUNTRIES = {"denmark": ("Denmark", "DNK"), "belgium": ("Belgium", "BEL"), "finland": ("Finland", "FIN")}

r = requests.get(NE_URL, timeout=300); r.raise_for_status()
with open("/tmp/ne_10m_admin_0_countries.zip", "wb") as f:
    f.write(r.content)
ne = gpd.read_file("zip:///tmp/ne_10m_admin_0_countries.zip")

for key, (admin, iso3) in COUNTRIES.items():
    land = ne[ne.ADMIN == admin][["ADMIN", "NAME", "ISO_A2", "ISO_A3", "geometry"]]
    assert len(land) == 1, (admin, len(land))
    os.makedirs(f"{HERE}/{key}", exist_ok=True)
    land.to_file(f"{HERE}/{key}/{key}_land.geojson", driver="GeoJSON")

    params = {"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": "MarineRegions:eez",
              "outputFormat": "application/json", "CQL_FILTER": f"iso_ter1='{iso3}' AND pol_type='200NM'"}
    w = requests.get(WFS, params=params, timeout=300); w.raise_for_status()
    eez = gpd.read_file(io.BytesIO(w.content))
    assert len(eez) == 1, (iso3, len(eez))
    eez.to_file(f"{HERE}/{key}/{key}_eez.geojson", driver="GeoJSON")
    print(f"{key}: land bounds {land.total_bounds.round(2)}, EEZ {eez.geoname.iloc[0]}, {eez.area_km2.iloc[0]} km2, bounds {eez.total_bounds.round(2)}")
