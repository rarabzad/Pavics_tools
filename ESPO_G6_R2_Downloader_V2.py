# Step 1: Import required libraries
import os, re, glob, shutil, time, json, zipfile, urllib.request, requests
from itertools import product
import numpy as np, pandas as pd, geopandas as gpd, xarray as xr
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from pyproj import CRS
import rasterio, rioxarray as rio
import cartopy.crs as ccrs
from siphon.catalog import TDSCatalog
from birdy import WPSClient
import ravenpy
from ravenpy.utilities.testdata import get_file
from rasterio.enums import Resampling

# Step 2: Define a helper template for RVT forcing inputs and replace placeholders
template_rvt = """\
:GriddedForcing            Precipitation
    :ForcingType           PRECIP
    :FileNameNC            forcing_file_path
    :VarNameNC             precipitation
    :DimNamesNC            rlon rlat time
    :RainCorrection        1
    :SnowCorrection        1
    :ElevationVarNameNC    elevation
    :RedirectToFile        GridWeights.txt
:EndGriddedForcing
:GriddedForcing            Maxtemp
    :ForcingType           TEMP_DAILY_MAX
    :FileNameNC            forcing_file_path
    :VarNameNC             max_temperature
    :DimNamesNC            rlon rlat time
    :TemperatureCorrection 1
    :ElevationVarNameNC    elevation
    :RedirectToFile        GridWeights.txt
:EndGriddedForcing
:GriddedForcing            Mintemp
    :ForcingType           TEMP_DAILY_MIN
    :FileNameNC            forcing_file_path
    :VarNameNC             min_temperature
    :DimNamesNC            rlon rlat time
    :TemperatureCorrection 1
    :ElevationVarNameNC    elevation
    :RedirectToFile        GridWeights.txt
:EndGriddedForcing"""
template_rvt = template_rvt.splitlines()

# Step 3: Replace file path placeholder with actual model output filename
output_file = f"Raven_input_{model_name}_{scenario}.nc"
template_rvt = [line.replace("forcing_file_path", output_file) for line in template_rvt]

# Step 4: Write updated RVT content to temporary file
with open("tmp.rvt", "w") as file:
    for line in template_rvt:
        file.write(line + "\n")

# Step 5: Load HRU shapefile and convert CRS to WGS84
hru = gpd.read_file(hrufile_path)
hru = hru.to_crs(epsg=4326)

# Step 6: Access remote climate data catalog using Siphon
url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"
cat = TDSCatalog(url)

# Step 7: Identify the dataset that matches the model and scenario
datasets = [dataset for dataset in cat.datasets]
id = [idx for idx, dataset in enumerate(cat.datasets) if model_name in dataset and scenario in dataset]
if not id:
    raise ValueError(f"No matching dataset found for model: {model_name}, scenario: {scenario}")
cds = cat.datasets[id[0]]

# Step 8: Open dataset with xarray and enable chunking
ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")

# Step 9: Define bounding box around HRU to select relevant data grid
minx, miny, maxx, maxy = hru.total_bounds + np.array([-0.06, -0.06, 0.06, 0.06])
bounding_box = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])

# Step 10: Identify grid points within the bounding box
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]
lat_within_bbox = (lat >= miny) & (lat <= maxy)
lon_within_bbox = (lon >= minx) & (lon <= maxx)
points_within_bbox = np.where(lat_within_bbox & lon_within_bbox)
rlat_ids, rlon_ids = points_within_bbox

# Step 11: Find bounding index range of latitude and longitude
row_indices = range(rlat_ids.min(), rlat_ids.max() + 1)
col_indices = range(rlon_ids.min(), rlon_ids.max() + 1)
lat_idx_min = np.min(row_indices)
lat_idx_max = np.max(row_indices)+1
lon_idx_min = np.min(col_indices)
lon_idx_max = np.max(col_indices)+1

# Step 12: Extract climate variables within bounding box
tasmin_values = ds.tasmin.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
tasmax_values = ds.tasmax.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
prcp_values = ds.pr.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
lat_values = ds.lat.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
lon_values = ds.lon.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))

# Step 13: Export bounding polygon of extracted data as GeoJSON
points = gpd.GeoDataFrame(
    {"geometry": [Point(lon, lat) for lon, lat in zip(lon_values.values.flatten(), lat_values.values.flatten())]},
    crs="EPSG:4326",
)
polygon = points.unary_union.convex_hull
buffered_polygon = polygon.buffer(0.06)
buffered_polygon_gdf = gpd.GeoDataFrame({"geometry": [buffered_polygon]}, crs="EPSG:4326")
buffered_polygon_gdf.to_file("bounding_box.geojson", driver="GeoJSON")

# Step 14: Load all extracted data into memory using Dask
tasmin_values = tasmin_values.compute()
tasmax_values = tasmax_values.compute()
prcp_values = prcp_values.compute()
lat_values = lat_values.compute()
lon_values = lon_values.compute()

# Step 15: Use WPSClient to get elevation data over selected region
url = os.environ.get("WPS_URL", "https://pavics.ouranos.ca/twitcher/ows/proxy/raven/wps")
geojson_file_path = os.path.join(os.getcwd(), "bounding_box.geojson")
wps = WPSClient(url)

# Step 16: Fetch and reproject elevation data, then resample
feature_url = get_file(geojson_file_path)
terrain_resp = wps.terrain_analysis(shape=feature_url, select_all_touching=True, projected_crs=3978)
properties, dem = terrain_resp.get(asobj=True)
dem_latlon = dem.rio.reproject(CRS.from_epsg(4326))

new_resolution = 0.05
dem_resampled = dem_latlon.rio.reproject(
    dem_latlon.rio.crs,
    resolution=(new_resolution, new_resolution),
    resampling=Resampling.bilinear
)
dem_resampled = dem_resampled.sortby(['x', 'y'])

# Step 17: Extract elevation values at each grid point
flat_lon = lon_values.values.flatten()
flat_lat = lat_values.values.flatten()
extracted_values = []
for lon, lat in zip(flat_lon, flat_lat):
    value = dem_resampled.sel(x=lon, y=lat, method="nearest").values
    extracted_values.append(value)
elevation_values = np.array(extracted_values).reshape(lat_values.values.shape)

# Step 18: Create a new xarray dataset for Raven with all variables
new_ds = xr.Dataset(
    {
        'min_temperature': (['time', 'rlat', 'rlon'], tasmin_values.values - 273.15),
        'max_temperature': (['time', 'rlat', 'rlon'], tasmax_values.values - 273.15),
        'precipitation':   (['time', 'rlat', 'rlon'], prcp_values.values * 86400),
        'lat':             (['rlat', 'rlon'],         lat_values.values),
        'lon':             (['rlat', 'rlon'],         lon_values.values),
        'elevation':       (['rlat', 'rlon'],         elevation_values),
    },
    coords={
        'time': ds['time'],
        'rlat': ds['rlat'][lat_idx_min:lat_idx_max],
        'rlon': ds['rlon'][lon_idx_min:lon_idx_max],
    }
)

# Step 19: Set units/metadata for each variable
new_ds.min_temperature.attrs['units'] = "degC"
new_ds.max_temperature.attrs['units'] = "degC"
new_ds.precipitation.attrs['units'] = "mm"
new_ds.lat.attrs['units'] = "degrees_north"
new_ds.lon.attrs['units'] = "degrees_east"
new_ds.elevation.attrs['units'] = "m"

# Step 20: Save processed dataset to NetCDF file
nc_file_path = os.path.join(os.getcwd(), output_file)
new_ds.to_netcdf(nc_file_path)

# Step 21: Copy the HRU shapefile components to working directory
shapefile_base = os.path.splitext(os.path.basename(hrufile_path))[0]
files_to_copy = glob.glob(os.path.join(os.path.dirname(hrufile_path), shapefile_base + '*'))
for file in files_to_copy:
    shutil.copy(file, os.getcwd())

# Step 22: Download and run grid weights generator script
script_filename = "derive_grid_weights.py"
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/julemai/GridWeightsGenerator/refs/heads/main/derive_grid_weights.py",
    script_filename
)
grid_weights_generator_path = script_filename
nc_path = output_file
shapefile_path = os.path.join(shapefile_base + ".shp")
dimname = "rlat,rlon"
varname = "lon,lat"
grid_weights_path = "GridWeights.txt"
command = (
    f'python {grid_weights_generator_path} '
    f'-i {nc_path} '
    f'-d "{dimname}" '
    f'-v "{varname}" '
    f'-r {shapefile_path} '
    f'-a --doall '
    f'-c {HRU_ID} '
    f'-o {grid_weights_path}'
)
os.system(command + " > /dev/null 2>&1")
