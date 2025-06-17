def ESPO_G6_R2_Downloader(model_name, scenario, hrufile_path, HRU_ID):
    import os, re, glob, shutil, json, zipfile, urllib.request
    import numpy as np, pandas as pd, geopandas as gpd, xarray as xr
    from shapely.geometry import Point, Polygon
    from pyproj import CRS
    import rioxarray as rio
    from siphon.catalog import TDSCatalog
    from birdy import WPSClient
    from ravenpy.utilities.testdata import get_file
    from rasterio.enums import Resampling

    # Step 1: Prepare the RVT template file with correct model/scenario naming
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
    output_file = f"Raven_input_{model_name}_{scenario}.nc"
    template_rvt = [line.replace("forcing_file_path", output_file) for line in template_rvt]
    with open("tmp.rvt", "w") as file:
        for line in template_rvt:
            file.write(line + "\n")

    # Step 2: Load and reproject HRU shapefile
    hru = gpd.read_file(hrufile_path)
    hru = hru.to_crs(epsg=4326)

    # Step 3: Access remote ESPO-G6-R2 climate data via Siphon
    url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"
    cat = TDSCatalog(url)

    # Step 4: Select matching dataset
    datasets = [dataset for dataset in cat.datasets]
    id = [idx for idx, dataset in enumerate(cat.datasets) if model_name in dataset and scenario in dataset]
    if not id:
        raise ValueError(f"No matching dataset found for model: {model_name}, scenario: {scenario}")
    cds = cat.datasets[id[0]]

    # Step 5: Open dataset with xarray and define bounding box
    ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")
    minx, miny, maxx, maxy = hru.total_bounds + np.array([-0.06, -0.06, 0.06, 0.06])
    bounding_box = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])

    # Step 6: Identify relevant grid cells
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    lat_within_bbox = (lat >= miny) & (lat <= maxy)
    lon_within_bbox = (lon >= minx) & (lon <= maxx)
    points_within_bbox = np.where(lat_within_bbox & lon_within_bbox)
    rlat_ids, rlon_ids = points_within_bbox
    lat_idx_min, lat_idx_max = np.min(rlat_ids), np.max(rlat_ids) + 1
    lon_idx_min, lon_idx_max = np.min(rlon_ids), np.max(rlon_ids) + 1

    # Step 7: Subset the climate data
    tasmin_values = ds.tasmin.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    tasmax_values = ds.tasmax.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    prcp_values = ds.pr.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lat_values = ds.lat.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lon_values = ds.lon.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))

    # Step 8: Save polygon and bounding box for elevation processing
    points = gpd.GeoDataFrame(
        {"geometry": [Point(lon, lat) for lon, lat in zip(lon_values.values.flatten(), lat_values.values.flatten())]},
        crs="EPSG:4326",
    )
    polygon = points.unary_union.convex_hull.buffer(0.06)
    buffered_polygon_gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")
    buffered_polygon_gdf.to_file("bounding_box.geojson", driver="GeoJSON")

    # Step 9: Trigger computation for climate variables
    tasmin_values = tasmin_values.compute()
    tasmax_values = tasmax_values.compute()
    prcp_values = prcp_values.compute()
    lat_values = lat_values.compute()
    lon_values = lon_values.compute()

    # Step 10: Get elevation data from WPS service
    wps = WPSClient(os.environ.get("WPS_URL", "https://pavics.ouranos.ca/twitcher/ows/proxy/raven/wps"))
    feature_url = get_file("bounding_box.geojson")
    terrain_resp = wps.terrain_analysis(shape=feature_url, select_all_touching=True, projected_crs=3978)
    properties, dem = terrain_resp.get(asobj=True)
    dem_latlon = dem.rio.reproject(CRS.from_epsg(4326))
    dem_resampled = dem_latlon.rio.reproject(
        dem_latlon.rio.crs,
        resolution=(0.05, 0.05),
        resampling=Resampling.bilinear,
    ).sortby(['x', 'y'])

    # Step 11: Extract elevation values for the grid
    flat_lon = lon_values.values.flatten()
    flat_lat = lat_values.values.flatten()
    elevation_values = np.array([
        dem_resampled.sel(x=lon, y=lat, method="nearest").values
        for lon, lat in zip(flat_lon, flat_lat)
    ]).reshape(lat_values.shape)

    # Step 12: Create a new NetCDF dataset with converted units
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

    new_ds.min_temperature.attrs['units'] = "degC"
    new_ds.max_temperature.attrs['units'] = "degC"
    new_ds.precipitation.attrs['units'] = "mm"
    new_ds.lat.attrs['units'] = "degrees_north"
    new_ds.lon.attrs['units'] = "degrees_east"
    new_ds.elevation.attrs['units'] = "m"

    # Step 13: Write NetCDF file
    new_ds.to_netcdf(output_file)

    # Step 14: Copy HRU shapefile files
    shapefile_base = os.path.splitext(os.path.basename(hrufile_path))[0]
    files_to_copy = glob.glob(os.path.join(os.path.dirname(hrufile_path), shapefile_base + '*'))
    for file in files_to_copy:
        shutil.copy(file, os.getcwd())

    # Step 15: Download and run grid weights generator
    script_filename = "derive_grid_weights.py"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/julemai/GridWeightsGenerator/refs/heads/main/derive_grid_weights.py",
        script_filename
    )
    command = (
        f'python {script_filename} '
        f'-i {output_file} '
        f'-d "rlat,rlon" '
        f'-v "lon,lat" '
        f'-r {shapefile_base}.shp '
        f'-a --doall '
        f'-c {HRU_ID} '
        f'-o GridWeights.txt'
    )
    os.system(command + " > /dev/null 2>&1")

    print(f"Finished generating forcing file: {output_file}")
