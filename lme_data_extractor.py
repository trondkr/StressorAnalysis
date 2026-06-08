"""
LME Data Extractor
==================
Extract LME-specific data from larger datasets (e.g., combined_test_NOAA)
when LME-specific data doesn't exist.

Uses LME shapefiles to clip data and saves results to GCS.

Author: Trond Kristiansen, ACTEA Inc.
Date: December 2025
"""

import sys
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import mapping
from typing import Optional, Tuple

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from ACTEA_gcs import ACTEA_gcs
except ImportError:
    print("Warning: Could not import ACTEA_gcs. Upload functionality will not work.")


class LMEDataExtractor:
    """
    Extract data for specific LME regions from larger datasets.

    Uses LME shapefiles to clip and save region-specific data.
    """

    def __init__(self, lme_number: int, shapefile_path: str = "gs://actea-shared/Shapefiles/LME66/LMEs66.shp"):
        """
        Initialize LME data extractor.

        Parameters
        ----------
        lme_number : int
            LME number to extract (e.g., 1 for East Bering Sea)
        shapefile_path : str
            Path to LME shapefile (can be GCS path)
        """
        self.lme_number = lme_number
        self.shapefile_path = shapefile_path
        self.gdf = None
        self.lme_geometry = None

    def load_lme_shapefile(self):
        """Load LME shapefile and extract geometry for specified LME."""
        print(f"\n  Loading LME shapefile: {self.shapefile_path}")

        try:
            self.gdf = gpd.read_file(self.shapefile_path)
            print(f"  ✓ Loaded shapefile with {len(self.gdf)} LME regions")

            # Find the LME number in the shapefile
            # Common column names: 'LME_NUMBER', 'LME_NUM', 'LME', 'LME_ID'
            lme_col = None
            for col in ['LME_NUMBER', 'LME_NUM', 'LME', 'LME_ID', 'lme_number']:
                if col in self.gdf.columns:
                    lme_col = col
                    break

            if lme_col is None:
                print(f"  ⚠ Warning: Could not find LME number column in shapefile")
                print(f"    Available columns: {list(self.gdf.columns)}")
                print(f"    Using first geometry as fallback")
                self.lme_geometry = self.gdf.iloc[0:1]
            else:
                # Filter to specified LME
                self.lme_geometry = self.gdf[self.gdf[lme_col] == self.lme_number]

                if len(self.lme_geometry) == 0:
                    print(f"  ✗ ERROR: LME {self.lme_number} not found in shapefile")
                    print(f"    Available LMEs: {sorted(self.gdf[lme_col].unique())}")
                    return False

                print(f"  ✓ Found LME {self.lme_number} geometry")

                # Print LME name if available
                if 'LME_NAME' in self.gdf.columns:
                    lme_name = self.lme_geometry['LME_NAME'].values[0]
                    print(f"    Name: {lme_name}")

            return True

        except Exception as e:
            print(f"  ✗ ERROR loading shapefile: {e}")
            return False

    def extract_lme_data(self, ds: xr.Dataset, var_name: str = None) -> Optional[xr.Dataset]:
        """
        Extract LME-specific data from dataset using shapefile clipping.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to clip
        var_name : str, optional
            Specific variable name to extract (if dataset has multiple)

        Returns
        -------
        xr.Dataset or None
            Clipped dataset for LME region
        """
        if self.lme_geometry is None:
            print("  ✗ ERROR: LME geometry not loaded. Call load_lme_shapefile() first.")
            return None

        print(f"\n  Clipping data to LME {self.lme_number} boundaries...")

        try:
            # Ensure dataset has spatial dims set
            if 'lon' in ds.dims and 'lat' in ds.dims:
                ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            elif 'longitude' in ds.dims and 'latitude' in ds.dims:
                ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            else:
                print(f"  ✗ ERROR: Could not find lat/lon dimensions in dataset")
                print(f"    Available dimensions: {list(ds.dims)}")
                return None

            # Set CRS (assume WGS84 for ocean data)
            ds.rio.write_crs("epsg:4326", inplace=True)

            # Get LME and dataset bounds for comparison
            lme_bounds = self.lme_geometry.total_bounds
            lme_lon_range = (lme_bounds[0], lme_bounds[2])
            lme_lat_range = (lme_bounds[1], lme_bounds[3])

            if 'lon' in ds.dims:
                ds_lon_range = (float(ds.lon.min()), float(ds.lon.max()))
                ds_lat_range = (float(ds.lat.min()), float(ds.lat.max()))
            else:
                ds_lon_range = (float(ds.longitude.min()), float(ds.longitude.max()))
                ds_lat_range = (float(ds.latitude.min()), float(ds.latitude.max()))

            print(f"  LME bounds: lon [{lme_lon_range[0]:.2f}, {lme_lon_range[1]:.2f}], lat [{lme_lat_range[0]:.2f}, {lme_lat_range[1]:.2f}]")
            print(f"  Dataset bounds: lon [{ds_lon_range[0]:.2f}, {ds_lon_range[1]:.2f}], lat [{ds_lat_range[0]:.2f}, {ds_lat_range[1]:.2f}]")

            # Check for overlap
            lon_overlap = not (lme_lon_range[1] < ds_lon_range[0] or lme_lon_range[0] > ds_lon_range[1])
            lat_overlap = not (lme_lat_range[1] < ds_lat_range[0] or lme_lat_range[0] > ds_lat_range[1])

            if not (lon_overlap and lat_overlap):
                print(f"  ✗ ERROR: LME {self.lme_number} is outside dataset geographic bounds")
                print(f"    No overlap between LME and dataset coverage")
                return None

            # Separate spatial and non-spatial variables
            # Clip only works on variables with lat/lon dimensions
            spatial_vars = []
            non_spatial_vars = {}

            for var in ds.data_vars:
                if 'lat' in ds[var].dims or 'latitude' in ds[var].dims:
                    spatial_vars.append(var)
                else:
                    # Save non-spatial variables to add back later
                    non_spatial_vars[var] = ds[var]

            # Create dataset with only spatial variables for clipping
            if spatial_vars:
                ds_spatial = ds[spatial_vars]
            else:
                print(f"  ✗ ERROR: No spatial variables found to clip")
                return None

            # Clip the spatial data to LME boundaries
            try:
                clipped_ds = ds_spatial.rio.clip(
                    self.lme_geometry.geometry.apply(mapping),
                    self.lme_geometry.crs,
                    drop=True,
                    all_touched=True  # Include cells that touch the boundary
                )

                # Add back non-spatial variables (like depth, time bounds, etc.)
                for var_name, var_data in non_spatial_vars.items():
                    clipped_ds[var_name] = var_data

            except Exception as clip_error:
                # Handle NoDataInBounds specifically
                if 'NoDataInBounds' in str(type(clip_error).__name__):
                    print(f"  ✗ ERROR: LME {self.lme_number} has no data overlap with dataset")
                    print(f"    The LME boundaries do not intersect with the dataset coverage.")
                    print(f"    This dataset does not cover this geographic region.")
                else:
                    print(f"  ✗ ERROR during clipping: {clip_error}")
                return None

            # Get bounds for reporting
            if hasattr(clipped_ds, 'lon') and hasattr(clipped_ds, 'lat'):
                lon_min, lon_max = float(clipped_ds.lon.min()), float(clipped_ds.lon.max())
                lat_min, lat_max = float(clipped_ds.lat.min()), float(clipped_ds.lat.max())
            elif hasattr(clipped_ds, 'longitude') and hasattr(clipped_ds, 'latitude'):
                lon_min, lon_max = float(clipped_ds.longitude.min()), float(clipped_ds.longitude.max())
                lat_min, lat_max = float(clipped_ds.latitude.min()), float(clipped_ds.latitude.max())
            else:
                lon_min, lon_max, lat_min, lat_max = 0, 0, 0, 0

            print(f"  ✓ Clipped to LME {self.lme_number}")
            print(f"    Bounds: lon [{lon_min:.2f}, {lon_max:.2f}], lat [{lat_min:.2f}, {lat_max:.2f}]")

            # Report data coverage
            if var_name and var_name in clipped_ds.data_vars:
                data_var = clipped_ds[var_name]
                total_cells = data_var.size
                valid_cells = np.sum(~np.isnan(data_var.values))
                print(f"    Valid data: {valid_cells:,}/{total_cells:,} cells ({valid_cells/total_cells*100:.1f}%)")

            return clipped_ds

        except Exception as e:
            print(f"  ✗ ERROR setting up clipping: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_to_gcs(self, ds: xr.Dataset, scenario: str, variable: str,
                    depth_type: str = 'bottom', depth_value: float = None):
        """
        Save extracted LME data to GCS.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to save
        scenario : str
            Climate scenario (e.g., 'ssp245')
        variable : str
            Variable name (e.g., 'thetao', 'o2', 'ph')
        depth_type : str
            'bottom' or 'surface'
        depth_value : float, optional
            Depth value for surface data (e.g., 5.0)
        """
        # Construct GCS path
        if depth_type == 'surface' and depth_value is not None:
            filename = f'{variable}_ensemble_sd+ba_surface_depth_{depth_value}_stats_{scenario}.nc'
        else:
            filename = f'{variable}_ensemble_sd+ba_{depth_type}_stats_{scenario}.nc'

        remote_path = f'downscaling/Arctic_LME_{self.lme_number}/{scenario}/ensemble/{variable}/{filename}'

        # Create local temporary file
        local_path = Path(f'/tmp/{filename}')

        print(f"\n  Saving extracted data to GCS...")
        print(f"    Remote: gs://actea-shared/{remote_path}")

        try:
            # Save locally first
            print(f"    Creating local file: {local_path}")
            ds.to_netcdf(local_path)

            # Upload to GCS
            print(f"    Uploading to GCS...")
            gcs = ACTEA_gcs(frequency='monthly')
            gcs.upload_to_gcs(str(local_path), to_fname=remote_path)

            # Clean up local file
            local_path.unlink()

            print(f"    ✓ Saved to: gs://actea-shared/{remote_path}")

            return remote_path

        except Exception as e:
            print(f"    ✗ ERROR saving to GCS: {e}")
            import traceback
            traceback.print_exc()

            # Clean up on error
            if local_path.exists():
                local_path.unlink()

            return None


def extract_lme_from_project(
    lme_number: int,
    source_project: str,
    scenario: str,
    variables: list = ['thetao', 'o2', 'ph'],
    surface_depth: float = 5.0,
    fallback_project: str = 'Arctic_6_pixel'
) -> dict:
    """
    Extract LME data from a source project dataset.

    Parameters
    ----------
    lme_number : int
        LME number to extract
    source_project : str
        Source project name (e.g., 'combined_test_NOAA')
    scenario : str
        Climate scenario
    variables : list
        Variables to extract
    surface_depth : float
        Surface depth value
    fallback_project : str
        Fallback project to use if source_project data is not available
        (default: 'Arctic_6_pixel')

    Returns
    -------
    dict
        Dictionary with extraction results
    """
    print("\n" + "="*80)
    print(f"EXTRACTING LME {lme_number} DATA FROM {source_project.upper()}")
    print(f"  Fallback dataset: {fallback_project.upper()}")
    print("="*80)

    extractor = LMEDataExtractor(lme_number)

    # Load shapefile
    if not extractor.load_lme_shapefile():
        return {'success': False, 'error': 'Failed to load shapefile'}

    # Initialize GCS
    gcs = ACTEA_gcs(frequency='monthly')

    results = {}

    # Extract each variable (surface and bottom)
    for var_name in variables:
        print(f"\n{'='*80}")
        print(f"Processing variable: {var_name}")
        print(f"{'='*80}")

        # Surface data
        print(f"\n[1/2] Surface data (depth={surface_depth}m)")
        surface_path = f'downscaling/{source_project}/{scenario}/ensemble/{var_name}/{var_name}_ensemble_sd+ba_surface_depth_{surface_depth}_stats_{scenario}.nc'
        fallback_surface_path = f'downscaling/{fallback_project}/{scenario}/ensemble/{var_name}/{var_name}_ensemble_sd+ba_surface_depth_{surface_depth}_stats_{scenario}.nc'

        try:
            print(f"  Attempting primary source: {surface_path}")
            ds_surface = gcs.open_dataset_on_gs(surface_path, decode_times=True)
            used_fallback_surface = False
            ds_surface_lme = None

            if ds_surface is None and fallback_project:
                print(f"  ⚠ Primary source not available, trying fallback: {fallback_surface_path}")
                ds_surface = gcs.open_dataset_on_gs(fallback_surface_path, decode_times=True)
                if ds_surface is not None:
                    print(f"  ✓ Successfully loaded from fallback dataset: {fallback_project}")
                    used_fallback_surface = True

            if ds_surface is not None:
                # Extract LME region
                ds_surface_lme = extractor.extract_lme_data(ds_surface, var_name=var_name)

                # If extraction failed due to geographic mismatch and we have a fallback, try it
                if ds_surface_lme is None and not used_fallback_surface and fallback_project:
                    print(f"  ⚠ Primary source has no geographic overlap with LME {lme_number}")
                    print(f"  ⚠ Trying fallback dataset: {fallback_surface_path}")
                    ds_surface_fallback = gcs.open_dataset_on_gs(fallback_surface_path, decode_times=True)
                    if ds_surface_fallback is not None:
                        print(f"  ✓ Successfully loaded from fallback dataset: {fallback_project}")
                        ds_surface_lme = extractor.extract_lme_data(ds_surface_fallback, var_name=var_name)
                        if ds_surface_lme is not None:
                            used_fallback_surface = True

                if ds_surface_lme is not None:
                    # Save to GCS
                    saved_path = extractor.save_to_gcs(
                        ds_surface_lme, scenario, var_name,
                        depth_type='surface', depth_value=surface_depth
                    )
                    results[f'{var_name}_surface'] = {
                        'success': True,
                        'path': saved_path,
                        'source': fallback_project if used_fallback_surface else source_project
                    }
                else:
                    results[f'{var_name}_surface'] = {
                        'success': False,
                        'error': 'Extraction failed - no geographic overlap with either dataset'
                    }
            else:
                print(f"  ✗ Could not load surface data from either primary or fallback source")
                results[f'{var_name}_surface'] = {
                    'success': False,
                    'error': 'Load failed from both sources'
                }

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results[f'{var_name}_surface'] = {
                'success': False,
                'error': str(e)
            }

        # Bottom data
        print(f"\n[2/2] Bottom data")
        bottom_path = f'downscaling/{source_project}/{scenario}/ensemble/{var_name}/{var_name}_ensemble_sd+ba_bottom_stats_{scenario}.nc'
        fallback_bottom_path = f'downscaling/{fallback_project}/{scenario}/ensemble/{var_name}/{var_name}_ensemble_sd+ba_bottom_stats_{scenario}.nc'

        try:
            print(f"  Attempting primary source: {bottom_path}")
            ds_bottom = gcs.open_dataset_on_gs(bottom_path, decode_times=True)
            used_fallback_bottom = False
            ds_bottom_lme = None

            if ds_bottom is None and fallback_project:
                print(f"  ⚠ Primary source not available, trying fallback: {fallback_bottom_path}")
                ds_bottom = gcs.open_dataset_on_gs(fallback_bottom_path, decode_times=True)
                if ds_bottom is not None:
                    print(f"  ✓ Successfully loaded from fallback dataset: {fallback_project}")
                    used_fallback_bottom = True

            if ds_bottom is not None:
                # Extract LME region
                ds_bottom_lme = extractor.extract_lme_data(ds_bottom, var_name=var_name)

                # If extraction failed due to geographic mismatch and we have a fallback, try it
                if ds_bottom_lme is None and not used_fallback_bottom and fallback_project:
                    print(f"  ⚠ Primary source has no geographic overlap with LME {lme_number}")
                    print(f"  ⚠ Trying fallback dataset: {fallback_bottom_path}")
                    ds_bottom_fallback = gcs.open_dataset_on_gs(fallback_bottom_path, decode_times=True)
                    if ds_bottom_fallback is not None:
                        print(f"  ✓ Successfully loaded from fallback dataset: {fallback_project}")
                        ds_bottom_lme = extractor.extract_lme_data(ds_bottom_fallback, var_name=var_name)
                        if ds_bottom_lme is not None:
                            used_fallback_bottom = True

                if ds_bottom_lme is not None:
                    # Save to GCS
                    saved_path = extractor.save_to_gcs(
                        ds_bottom_lme, scenario, var_name,
                        depth_type='bottom'
                    )
                    results[f'{var_name}_bottom'] = {
                        'success': True,
                        'path': saved_path,
                        'source': fallback_project if used_fallback_bottom else source_project
                    }
                else:
                    results[f'{var_name}_bottom'] = {
                        'success': False,
                        'error': 'Extraction failed - no geographic overlap with either dataset'
                    }
            else:
                print(f"  ✗ Could not load bottom data from either primary or fallback source")
                results[f'{var_name}_bottom'] = {
                    'success': False,
                    'error': 'Load failed from both sources'
                }

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results[f'{var_name}_bottom'] = {
                'success': False,
                'error': str(e)
            }

    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)

    successful = [k for k, v in results.items() if v.get('success', False)]
    failed = [k for k, v in results.items() if not v.get('success', False)]

    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    for key in successful:
        print(f"    • {key}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for key in failed:
            error = results[key].get('error', 'Unknown error')
            print(f"    • {key}: {error}")

    results['summary'] = {
        'total': len(results) - 1,  # -1 to exclude summary itself
        'successful': len(successful),
        'failed': len(failed)
    }

    return results


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Extract LME data from larger datasets')
    parser.add_argument('--lme', type=int, required=True, help='LME number (e.g., 1 for East Bering Sea)')
    parser.add_argument('--source', default='combined_test_NOAA', help='Source project')
    parser.add_argument('--scenario', default='ssp245', help='Climate scenario')
    parser.add_argument('--variables', nargs='+', default=['thetao', 'o2', 'ph'], help='Variables to extract')

    args = parser.parse_args()

    results = extract_lme_from_project(
        lme_number=args.lme,
        source_project=args.source,
        scenario=args.scenario,
        variables=args.variables
    )

    print(f"\n{'='*80}")
    print(f"DONE! Extracted {results['summary']['successful']} datasets for LME {args.lme}")
    print(f"{'='*80}")
