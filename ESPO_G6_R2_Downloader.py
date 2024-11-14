def ESPO_G6_R2_Downloader(shapefile_path, model_name, scenario):
    # Step 1: Load the Shapefile (Region of interest)
    region = gpd.read_file(shapefile_path)
    # Ensure the region is in the same CRS as the climate data (lat/lon, WGS84)
    region = region.to_crs(epsg=4326)
    # Get the min/max bounds of the region (lat/lon)
    min_lon, min_lat, max_lon, max_lat = region.total_bounds  # returns (minx, miny, maxx, maxy)
    # Step 2: Access climate data using Siphon
    url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"  # Change to your dataset URL
    cat = TDSCatalog(url)
    # Retrieve the datasets directly (these will be strings, not objects)
    datasets = [dataset for dataset in cat.datasets]  # cat.datasets contains strings, no need for .name
    # Find the dataset matching the model_name and scenario
    id = [idx for idx, dataset in enumerate(cat.datasets) if model_name in dataset and scenario in dataset]
    # If no matching dataset is found, skip the iteration
    if not id:
        print(f"No matching dataset found for model: {model_name}, scenario: {scenario}")
        return
    cds = cat.datasets[id[0]]
    # Open the dataset using xarray (this will use Dask for chunking)
    ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")  # Adjust chunking as needed
    # Step 3: Find the corresponding grid indices for min/max lat/lon
    row_indices, col_indices = np.where((ds['lat'].values >= min_lat) & (ds['lat'].values <= max_lat)
    lat_idx_min = np.min(row_indices)
    lat_idx_max = np.max(row_indices)
    lon_idx_min = np.min(col_indices)
    lon_idx_max = np.max(col_indices)
    # Step 4: Extract the subgrid (box-constrained region) for the entire time range
    tasmin_values = ds.tasmin.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    tasmax_values = ds.tasmax.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    prcp_values = ds.pr.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lat_values = ds.lat.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lon_values = ds.lon.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    # Step 5: Compute the data (if necessary)
    tasmin_values = tasmin_values.compute()  # Trigger computation
    tasmax_values = tasmax_values.compute()  # Similarly for tasmax
    prcp_values = prcp_values.compute()  # Similarly for prcp
    # Step 6: Write the extracted data to a NetCDF file
    # Create a new NetCDF file with the same dimensions
    output_file = f"Raven_input_{model_name}_{scenario}.nc"
    # Create a new xarray Dataset for the output data
    new_ds = xr.Dataset(
        {
            'tasmin': (['time', 'rlat', 'rlon'], tasmin_values.values),  # Use the extracted tasmin values
            'tasmax': (['time', 'rlat', 'rlon'], tasmax_values.values),  # Use the extracted tasmax values
            'prcp': (['time', 'rlat', 'rlon'], prcp_values.values),      # Use the extracted prcp values
            'lat':    (['rlat', 'rlon'],         lat_values.values),     # Use the extracted lat values
            'lon':    (['rlat', 'rlon'],         lon_values.values),     # Use the extracted lon values
        },
        coords={
            'time': ds['time'],  # Time coordinate (same as in the original dataset)
            'rlat': ds['rlat'][lat_idx_min:lat_idx_max],  # Latitude coordinate (subset)
            'rlon': ds['rlon'][lon_idx_min:lon_idx_max],  # Longitude coordinate (subset)
        }
    )
    new_ds.tasmin.attrs['units'] = "degC"
    new_ds.tasmax.attrs['units'] = "degC"
    new_ds.prcp.attrs['units'] = "mm"
    new_ds.lat.attrs['units'] = "degrees_north"
    new_ds.lon.attrs['units'] = "degrees_east"
    # Save the output to NetCDF
    new_ds.to_netcdf(output_file)
    # Print success message
    print(f"File {output_file} written successfully.")
