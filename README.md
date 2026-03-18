# Scotland Railway Map Project

Interactive visualisation of Scottish railway stations and lines using multiple frameworks.

## Project Structure

```
├── apps/                           # Web applications
│   ├── flask_app.py                # Flask + Folium map (port 5000)
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── folium_map.ipynb            # Folium map with ODM data
│   ├── folium_flask_map.ipynb      # Flask + Folium notebook
│   └── plotly_popup_experiment.ipynb  # Plotly charts inside popups
├── data/                           # All data files
│   ├── stations.csv                # Scottish station data
│   ├── uk_railways2.geojson        # Railway line geometries
│   ├── scotland.json               # Scotland boundary GeoJSON
│   ├── Origination Destination Matrix.csv
│   └── ...
├── output/                         # Generated HTML map files
├── assets/icons/                   # Map marker icons (SVG)
├── templates/                      # Flask HTML templates
├── docs/                           # Documentation & exports
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the main Flask app
python apps/flask_app.py
```

## Data Sources
- Station data filtered for `constituentCountry == 'scotland'`
- Railway lines from OpenStreetMap via overpass-turbo (see docs/osm_license.txt)
