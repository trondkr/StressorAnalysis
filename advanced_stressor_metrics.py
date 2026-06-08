"""
Advanced Stressor Metrics Module
=================================
Implements sophisticated analyses for multi-stressor assessment including:
1. Compound Event Typology (simultaneous, sequential, cascading)
2. Recovery Time Analysis
3. Stressor Persistence Metrics
4. Rate of Change Analysis
5. Threshold Crossing Analysis
6. Spatial Refugia Stability
7. Cross-Shelf Gradient Analysis
10. Stressor Interaction Index

Based on suggestions from compound stressor analysis framework

Author: Trond Kristiansen
Date: November 2025
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from scipy import stats, signal
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedStressorMetrics:
    """
    Advanced metrics for multi-stressor oceanographic analysis
    """
    
    def __init__(self, baseline_period: Tuple[int, int] = (2017, 2022)):
        """
        Initialize advanced metrics calculator
        
        Parameters
        ----------
        baseline_period : tuple
            Start and end years for baseline (e.g., (2017, 2022))
        """
        self.baseline_period = baseline_period
        
    
    def classify_compound_events(
        self,
        extremes: Dict[str, xr.Dataset],
        temporal_window: int = 30,
        lag_threshold: int = 15,
        skip_sequential: bool = False,
        skip_cascading: bool = False,
        time_unit: str = 'day'
    ) -> Dict[str, xr.DataArray]:
        """
        SUGGESTION 1: Compound Event Typology and Evolution

        Classify events as:
        - Simultaneous: Multiple stressors occur at the same time
        - Sequential: One stressor follows another within temporal_window time steps
        - Cascading: One stressor appears to trigger another (sequential with increasing intensity)
        - Triple-threat: All three main stressors occur together

        Parameters
        ----------
        extremes : dict
            Dictionary of extreme event datasets from stressor detection
        temporal_window : int
            Time steps to consider events as "sequential" (default: 30 for daily data)
        lag_threshold : int
            Time steps for potential cascading detection (default: 15 for daily data)
        skip_sequential : bool
            Skip sequential event detection to save computation (default: False)
        skip_cascading : bool
            Skip cascading event detection to save computation (default: False)
        time_unit : str
            Unit of temporal_window and lag_threshold: 'day' or 'month' (default: 'day')

        Returns
        -------
        dict
            Classification results with counts and evolution metrics
        """
        print("\n" + "="*80)
        print("SUGGESTION 1: COMPOUND EVENT TYPOLOGY")
        print("="*80)
        
        results = {}
        
        # Convert to binary presence/absence
        binary_events = {}
        for name, data in extremes.items():
            # Use 'is_extreme' if available, otherwise threshold intensity
            if 'is_extreme' in data:
                binary_events[name] = data['is_extreme']
            elif 'intensity' in data:
                binary_events[name] = data['intensity'] > 0
            else:
                print(f"Warning: Cannot create binary for {name}")
                continue
        
        # SIMULTANEOUS EVENTS: Same time point
        print("\nDetecting simultaneous events...")
        if len(binary_events) >= 2:
            keys = list(binary_events.keys())

            # Helper function to align two DataArrays by time
            def align_by_time(da1, da2):
                if 'time' in da1.dims and 'time' in da2.dims:
                    if da1.sizes['time'] != da2.sizes['time']:
                        common_times = np.intersect1d(da1.time.values, da2.time.values)
                        if len(common_times) > 0:
                            return da1.sel(time=common_times), da2.sel(time=common_times)
                return da1, da2

            # Pairwise simultaneous events
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    key1, key2 = keys[i], keys[j]
                    e1, e2 = align_by_time(binary_events[key1], binary_events[key2])
                    simultaneous = e1 & e2

                    name = f"simultaneous_{key1}_{key2}"
                    results[name] = simultaneous

                    # Calculate frequency over time
                    if 'time' in simultaneous.dims:
                        yearly = simultaneous.groupby('time.year').sum('time')
                        results[f"{name}_yearly_count"] = yearly

            # Triple threat if we have 3+ stressors
            if len(keys) >= 3:
                # Find common times across all stressors
                common_times = None
                for key in keys:
                    if 'time' in binary_events[key].dims:
                        times = binary_events[key].time.values
                        if common_times is None:
                            common_times = times
                        else:
                            common_times = np.intersect1d(common_times, times)

                if common_times is not None and len(common_times) > 0:
                    aligned = [binary_events[key].sel(time=common_times) for key in keys]
                    triple = aligned[0].copy()
                    for arr in aligned[1:]:
                        triple = triple & arr
                else:
                    triple = binary_events[keys[0]].copy()
                    for key in keys[1:]:
                        triple = triple & binary_events[key]

                results['triple_threat'] = triple

                if 'time' in triple.dims:
                    results['triple_threat_yearly'] = triple.groupby('time.year').sum('time')
        
        # SEQUENTIAL EVENTS: Events within temporal window
        # Define scientifically meaningful sequences where the first event
        # is expected to precede/trigger the second event
        import gc
        if not skip_sequential:
            unit = 'month' if time_unit == 'month' else 'day'
            unit_str = f"{unit}s" if temporal_window != 1 else unit
            print(f"\nDetecting sequential events (window: {temporal_window} {unit_str})...")
            if 'time' in list(binary_events.values())[0].dims:
                # Define meaningful sequential pairs (driver -> response)
                # Scientific rationale:
                # - surface_heatwave -> bottom_heatwave: heat propagates downward
                # - surface_heatwave -> hypoxia: warm water holds less O2, increased respiration
                # - surface_heatwave -> acidification: affects CO2 solubility and biological activity
                # - bottom_heatwave -> hypoxia: warm bottom water holds less O2
                # - bottom_heatwave -> acidification: temperature affects carbonate chemistry
                # - hypoxia -> acidification: respiration produces CO2, lowering pH

                sequential_pairs = [
                    ('surface_heatwave', 'bottom_heatwave'),
                    ('surface_heatwave', 'hypoxia'),
                    ('surface_heatwave', 'acidification'),
                    ('bottom_heatwave', 'hypoxia'),
                    ('bottom_heatwave', 'acidification'),
                    ('hypoxia', 'acidification'),
                ]

                pair_count = 0
                for name1, name2 in sequential_pairs:
                    if name1 in binary_events and name2 in binary_events:
                        pair_count += 1
                        print(f"  Pair {pair_count}: {name1} -> {name2}")
                        sequential = self._detect_sequential(
                            binary_events[name1], binary_events[name2], temporal_window
                        )
                        results[f"sequential_{name1}_to_{name2}"] = sequential
                        gc.collect()  # Free memory after each pair

                if pair_count == 0:
                    print("  No matching stressor pairs found for sequential detection")
        else:
            print("\nSkipping sequential event detection (not needed for visualization)")

        # CASCADING EVENTS: Sequential with intensity increase
        # Use same scientifically meaningful pairs as sequential events
        if not skip_cascading:
            unit = 'month' if time_unit == 'month' else 'day'
            unit_str = f"{unit}s" if lag_threshold != 1 else unit
            print(f"\nDetecting cascading events (lag threshold: {lag_threshold} {unit_str})...")

            # Same pairs as sequential - driver triggers response with intensity increase
            cascading_pairs = [
                ('surface_heatwave', 'bottom_heatwave'),
                ('surface_heatwave', 'hypoxia'),
                ('surface_heatwave', 'acidification'),
                ('bottom_heatwave', 'hypoxia'),
                ('bottom_heatwave', 'acidification'),
                ('hypoxia', 'acidification'),
            ]

            pair_count = 0
            for name1, name2 in cascading_pairs:
                if name1 in extremes and name2 in extremes:
                    data1 = extremes[name1]
                    data2 = extremes[name2]

                    if 'intensity' in data1 and 'intensity' in data2:
                        pair_count += 1
                        print(f"  Pair {pair_count}: {name1} -> {name2}")
                        cascading = self._detect_cascading(
                            binary_events[name1],
                            data1['intensity'],
                            binary_events[name2],
                            data2['intensity'],
                            lag_threshold
                        )
                        results[f"cascading_{name1}_triggers_{name2}"] = cascading
                        gc.collect()  # Free memory after each pair

            if pair_count == 0:
                print("  No matching stressor pairs found for cascading detection")
        else:
            print("\nSkipping cascading event detection (not needed for visualization)")
        
        # EVOLUTION METRICS
        print("\nCalculating evolution of compound events...")
        if 'triple_threat' in results and 'time' in results['triple_threat'].dims:
            # Trend in triple-threat frequency
            yearly = results['triple_threat'].groupby('time.year').sum(dim='time')
            
            # Rechunk spatial dimensions to single chunks to avoid dask parallelization issues
            if hasattr(yearly.data, 'chunks'):
                # Rechunk lat and lon to single chunks (-1 means all in one chunk)
                yearly = yearly.chunk({'lat': -1, 'lon': -1}) if 'lat' in yearly.dims and 'lon' in yearly.dims else yearly
            
            # Calculate trend for each location
            def calc_trend(data):
                if len(data) < 3:
                    return np.nan
                years = np.arange(len(data))
                try:
                    slope, _, _, _, _ = stats.linregress(years, data)
                    return slope
                except:
                    return np.nan
            
            trend = xr.apply_ufunc(
                calc_trend,
                yearly,
                input_core_dims=[['year']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'allow_rechunk': True}
            )
            results['triple_threat_trend'] = trend
        
        print(f"\nClassified {len(results)} compound event types")
        return results
    
    
    def _detect_sequential(
        self,
        events1: xr.DataArray,
        events2: xr.DataArray,
        window: int
    ) -> xr.DataArray:
        """Helper to detect sequential events (memory optimized)"""
        import gc

        # Align time coordinates if they differ
        if 'time' in events1.dims and 'time' in events2.dims:
            if events1.sizes['time'] != events2.sizes['time']:
                print(f"    Aligning time dimensions: {events1.sizes['time']} vs {events2.sizes['time']}")
                # Find common time coordinates
                common_times = np.intersect1d(events1.time.values, events2.time.values)
                if len(common_times) == 0:
                    print(f"    WARNING: No overlapping time coordinates, skipping this pair")
                    return xr.DataArray(
                        np.zeros_like(events1.values, dtype=bool),
                        dims=events1.dims,
                        coords=events1.coords
                    )
                print(f"    Using {len(common_times)} common time steps")
                events1 = events1.sel(time=common_times)
                events2 = events2.sel(time=common_times)

        # Get dimensions
        time_dim = events1.sizes.get('time', 0)

        # For very large datasets, use a memory-optimized approach
        if time_dim > 5000:
            print(f"    Using memory-optimized approach for {time_dim} time steps...")

            # Work with numpy arrays to reduce memory overhead
            e1 = events1.values.astype(np.float32) if hasattr(events1, 'values') else np.asarray(events1, dtype=np.float32)
            e2 = events2.values.astype(np.float32) if hasattr(events2, 'values') else np.asarray(events2, dtype=np.float32)

            # Initialize result
            sequential_result = np.zeros_like(e1, dtype=bool)

            # Use larger lag steps to reduce iterations (sample every 5 days)
            step_size = max(1, window // 6)
            lags_to_check = list(range(1, window + 1, step_size))
            if window not in lags_to_check:
                lags_to_check.append(window)

            print(f"    Checking {len(lags_to_check)} representative lags: {lags_to_check}")

            for lag in lags_to_check:
                # Shift array manually
                shifted_e2 = np.roll(e2, lag, axis=0)
                # Zero out the rolled portion to avoid false positives
                shifted_e2[:lag] = 0

                # Sequential condition
                seq_mask = (e1 > 0) & (shifted_e2 > 0)
                sequential_result = sequential_result | seq_mask

                # Free memory
                del shifted_e2, seq_mask

            # Free input arrays
            del e1, e2
            gc.collect()

            # Convert back to xarray
            result = xr.DataArray(
                sequential_result,
                dims=events1.dims,
                coords=events1.coords
            )
            del sequential_result
            gc.collect()

            return result

        # Original approach for smaller datasets
        # Compute input arrays first to avoid building huge dask graphs
        if hasattr(events1, 'compute'):
            print(f"    Computing events1 for sequential detection...")
            events1 = events1.compute()
        if hasattr(events2, 'compute'):
            print(f"    Computing events2 for sequential detection...")
            events2 = events2.compute()

        # Roll the second event forward in time
        sequential = xr.zeros_like(events1)

        # Process in chunks to show progress and avoid memory issues
        chunk_size = 5
        for chunk_start in range(1, window + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, window + 1)
            print(f"    Processing lags {chunk_start}-{chunk_end-1} of {window}...")

            for lag in range(chunk_start, chunk_end):
                shifted = events2.roll(time=lag, roll_coords=False)
                # Where event1 occurred, then event2 within window
                sequential = sequential | (events1 & shifted)

        return sequential
    
    
    def _detect_cascading(
        self,
        events1: xr.DataArray,
        intensity1: xr.DataArray,
        events2: xr.DataArray,
        intensity2: xr.DataArray,
        lag_threshold: int
    ) -> xr.DataArray:
        """Helper to detect cascading events (intensity increase) - memory optimized"""
        import gc

        # Align time dimensions if they differ
        time_dim1 = events1.sizes.get('time', 0)
        time_dim2 = events2.sizes.get('time', 0)

        if time_dim1 != time_dim2:
            print(f"    Aligning time dimensions: events1={time_dim1}, events2={time_dim2}")
            # Find common time coordinates
            common_times = np.intersect1d(events1.time.values, events2.time.values)
            if len(common_times) == 0:
                print(f"    WARNING: No overlapping time steps found, returning zeros")
                return xr.zeros_like(events1, dtype=bool)

            print(f"    Using {len(common_times)} common time steps")
            # Select only common times
            events1 = events1.sel(time=common_times)
            intensity1 = intensity1.sel(time=common_times)
            events2 = events2.sel(time=common_times)
            intensity2 = intensity2.sel(time=common_times)

        # Get dimensions
        time_dim = events1.sizes.get('time', 0)

        # For very large datasets, use a simplified approach
        if time_dim > 5000:
            print(f"    Using memory-optimized approach for {time_dim} time steps...")

            # Work with numpy arrays to reduce memory overhead
            # Convert to boolean/float32 to save memory
            e1 = events1.values.astype(np.float32) if hasattr(events1, 'values') else np.asarray(events1, dtype=np.float32)
            i1 = intensity1.values.astype(np.float32) if hasattr(intensity1, 'values') else np.asarray(intensity1, dtype=np.float32)
            e2 = events2.values.astype(np.float32) if hasattr(events2, 'values') else np.asarray(events2, dtype=np.float32)
            i2 = intensity2.values.astype(np.float32) if hasattr(intensity2, 'values') else np.asarray(intensity2, dtype=np.float32)

            # Initialize result
            cascading_result = np.zeros_like(e1, dtype=bool)

            # Use larger lag steps to reduce iterations
            step_size = max(1, lag_threshold // 3)
            lags_to_check = list(range(1, lag_threshold + 1, step_size))
            if lag_threshold not in lags_to_check:
                lags_to_check.append(lag_threshold)

            print(f"    Checking {len(lags_to_check)} representative lags: {lags_to_check}")

            for lag in lags_to_check:
                # Shift arrays manually
                shifted_e2 = np.roll(e2, lag, axis=0)
                shifted_i2 = np.roll(i2, lag, axis=0)

                # Zero out the rolled portion to avoid false positives
                shifted_e2[:lag] = 0
                shifted_i2[:lag] = 0

                # Cascade condition
                cascade_mask = (e1 > 0) & (shifted_e2 > 0) & (shifted_i2 > i1)
                cascading_result = cascading_result | cascade_mask

                # Free memory
                del shifted_e2, shifted_i2, cascade_mask

            # Free input arrays
            del e1, i1, e2, i2
            gc.collect()

            # Convert back to xarray
            result = xr.DataArray(
                cascading_result,
                dims=events1.dims,
                coords=events1.coords
            )
            del cascading_result
            gc.collect()

            return result

        # Original approach for smaller datasets
        if hasattr(events1, 'compute'):
            print(f"    Computing events1 for cascading detection...")
            events1 = events1.compute()
        if hasattr(intensity1, 'compute'):
            print(f"    Computing intensity1 for cascading detection...")
            intensity1 = intensity1.compute()
        if hasattr(events2, 'compute'):
            print(f"    Computing events2 for cascading detection...")
            events2 = events2.compute()
        if hasattr(intensity2, 'compute'):
            print(f"    Computing intensity2 for cascading detection...")
            intensity2 = intensity2.compute()

        cascading = xr.zeros_like(events1)

        # Process in chunks to show progress
        chunk_size = 5
        for chunk_start in range(1, lag_threshold + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, lag_threshold + 1)
            print(f"    Processing lags {chunk_start}-{chunk_end-1} of {lag_threshold}...")

            # Look for cases where event2 follows event1 with higher intensity
            for lag in range(chunk_start, chunk_end):
                shifted_events = events2.roll(time=lag, roll_coords=False)
                shifted_intensity = intensity2.roll(time=lag, roll_coords=False)

                # Event1 present, then event2 with higher intensity
                cascade_condition = (
                    events1 &
                    shifted_events &
                    (shifted_intensity > intensity1)
                )
                cascading = cascading | cascade_condition

        return cascading
    
    
    def calculate_recovery_times(
        self,
        extremes: Dict[str, xr.Dataset],
        percentiles: List[float] = [25, 50, 75, 90]
    ) -> Dict[str, xr.DataArray]:
        """
        SUGGESTION 2: Recovery Time Analysis
        
        Calculate return intervals between extreme events and identify
        areas where recovery time is shrinking below critical thresholds
        
        Parameters
        ----------
        extremes : dict
            Dictionary of extreme event datasets
        percentiles : list
            Percentiles of return intervals to report
            
        Returns
        -------
        dict
            Return interval statistics and trends
        """
        print("\n" + "="*80)
        print("SUGGESTION 2: RECOVERY TIME ANALYSIS")
        print("="*80)
        
        results = {}
        
        for name, data in extremes.items():
            print(f"\nAnalyzing recovery times for {name}...")
            
            # Get binary extreme events
            if 'is_extreme' in data:
                events = data['is_extreme']
            elif 'intensity' in data:
                events = data['intensity'] > 0
            else:
                continue
            
            # Calculate return intervals for each location
          
            return_intervals = self._calculate_return_intervals(events)
            
            results[f"{name}_return_interval_mean"] = return_intervals['mean']
            results[f"{name}_return_interval_median"] = return_intervals['median']
            results[f"{name}_return_interval_min"] = return_intervals['min']
            
            # Calculate percentiles
            for p in percentiles:
                results[f"{name}_return_interval_p{p}"] = return_intervals[f'p{p}']
            
            # Trend in return intervals (are they getting shorter?)
            if 'time' in events.dims:
                trend = self._calculate_return_interval_trend(events)
                results[f"{name}_return_interval_trend"] = trend
                
                # Identify areas with critically short recovery
                # (return interval < 30 days for daily data, < 3 months for monthly)
                if self._is_daily_data(events):
                    critical_threshold = 30  # days
                else:
                    critical_threshold = 3  # months
                
                critical_areas = return_intervals['median'] < critical_threshold
                results[f"{name}_critical_recovery_areas"] = critical_areas
                
                print(f"  Mean return interval: {float(return_intervals['mean'].mean()):.1f} time steps")
                print(f"  Areas with critical recovery (<{critical_threshold}): "
                      f"{float(critical_areas.sum())} locations")
        
        return results
    
    
    def _calculate_return_intervals(self, events: xr.DataArray) -> Dict[str, xr.DataArray]:
        """Calculate return intervals between events"""
        
        def calc_intervals(event_series):
            """Calculate intervals for a single location time series"""
            if not isinstance(event_series, np.ndarray):
                event_series = np.array(event_series)
            
            # Find indices where events occur
            event_indices = np.where(event_series > 0)[0]
            
            if len(event_indices) < 2:
                return {
                    'mean': np.nan, 'median': np.nan, 'min': np.nan,
                    'p25': np.nan, 'p50': np.nan, 'p75': np.nan, 'p90': np.nan
                }
            
            # Calculate intervals between consecutive events
            intervals = np.diff(event_indices)
            
            return {
                'mean': np.mean(intervals),
                'median': np.median(intervals),
                'min': np.min(intervals),
                'p25': np.percentile(intervals, 25),
                'p50': np.percentile(intervals, 50),
                'p75': np.percentile(intervals, 75),
                'p90': np.percentile(intervals, 90)
            }
        
        # Apply to each location
        result_dict = {}
        for key in ['mean', 'median', 'min', 'p25', 'p50', 'p75', 'p90']:
            result_dict[key] = xr.apply_ufunc(
                lambda x: calc_intervals(x)[key],
                events,
                input_core_dims=[['time']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'allow_rechunk': True}
            )
        
        return result_dict
    
    
    def _calculate_return_interval_trend(self, events: xr.DataArray) -> xr.DataArray:
        """Calculate trend in return intervals over time"""
        
        def calc_trend(event_series):
            """Calculate trend for a single location"""
            event_indices = np.where(event_series > 0)[0]
            
            if len(event_indices) < 3:
                return np.nan
            
            intervals = np.diff(event_indices)
            time = np.arange(len(intervals))
            
            try:
                slope, _, _, _, _ = stats.linregress(time, intervals)
                return slope  # Negative = getting shorter (worse)
            except:
                return np.nan
        
        trend = xr.apply_ufunc(
            calc_trend,
            events,
            input_core_dims=[['time']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        
        return trend
    
    
    def calculate_persistence_metrics(
        self,
        extremes: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.DataArray]:
        """
        SUGGESTION 3: Stressor Persistence Metrics
        
        Calculate percentage of time under single, double, or triple
        stressor conditions. Identify chronic vs. episodic exposure patterns.
        
        Parameters
        ----------
        extremes : dict
            Dictionary of extreme event datasets
            
        Returns
        -------
        dict
            Persistence metrics including exposure percentages and chronicity indices
        """
        print("\n" + "="*80)
        print("SUGGESTION 3: STRESSOR PERSISTENCE METRICS")
        print("="*80)
        
        results = {}
        
        # Convert to binary events
        binary_events = {}
        for name, data in extremes.items():
            if 'is_extreme' in data:
                binary_events[name] = data['is_extreme']
            elif 'intensity' in data:
                binary_events[name] = (data['intensity'] > 0).astype(float)
        
        if len(binary_events) == 0:
            print("Warning: No binary events to analyze")
            return results
        
        # Calculate exposure percentages
        print("\nCalculating exposure percentages...")
        for name, events in binary_events.items():
            # Overall percentage of time exposed
            if 'time' in events.dims:
                pct_exposed = (events.sum('time') / events.sizes['time']) * 100
                results[f"{name}_percent_time_exposed"] = pct_exposed
                
                print(f"  {name}: {float(pct_exposed.mean()):.1f}% time exposed (spatial mean)")
        
        # Count simultaneous stressors
        print("\nCalculating multi-stressor exposure...")
        if len(binary_events) > 1:
            # Stack all events - drop non-essential coordinates to avoid conflicts
            event_list = []
            for events in binary_events.values():
                # Keep only essential dimensions and coordinates
                events_clean = events.drop_vars(
                    [v for v in events.coords if v not in events.dims],
                    errors='ignore'
                )
                event_list.append(events_clean)
            stacked = xr.concat(event_list, dim='stressor')
            
            # Count number of concurrent stressors
            n_concurrent = stacked.sum('stressor')
            
            # Percentage of time with 0, 1, 2, 3+ stressors
            total_time = n_concurrent.sizes.get('time', 1)
            
            results['no_stressor_percent'] = ((n_concurrent == 0).sum('time') / total_time * 100)
            results['single_stressor_percent'] = ((n_concurrent == 1).sum('time') / total_time * 100)
            results['double_stressor_percent'] = ((n_concurrent == 2).sum('time') / total_time * 100)
            
            if len(binary_events) >= 3:
                results['triple_stressor_percent'] = ((n_concurrent >= 3).sum('time') / total_time * 100)
        
        # Chronicity index: ratio of persistent to episodic exposure
        print("\nCalculating chronicity indices...")
        for name, events in binary_events.items():
            chronicity = self._calculate_chronicity(events)
            results[f"{name}_chronicity_index"] = chronicity
        
        # Year of incidental → chronic transition (first year when cumulative exposure exceeds threshold)
        # Use 10% threshold: 25% is rarely reached for episodic stressors; 10% captures meaningful chronic transition
        print("\nCalculating incidental-to-chronic transition year...")
        for name, events in binary_events.items():
            if 'time' in events.dims:
                trans_year = self._calculate_incidental_to_chronic_year(events, chronic_threshold=0.10)
                results[f"{name}_incidental_to_chronic_year"] = trans_year
        
        # Transition from seasonal to year-round exposure
        print("\nAnalyzing seasonal transitions...")
        for name, events in binary_events.items():
            if 'time' in events.dims:
                seasonal_coverage = self._calculate_seasonal_coverage(events)
                results[f"{name}_months_with_exposure"] = seasonal_coverage
        
        return results
    
    
    def _calculate_chronicity(self, events: xr.DataArray) -> xr.DataArray:
        """
        Calculate chronicity index: ratio of event duration to event frequency
        Higher values = fewer but longer events (chronic)
        Lower values = many brief events (episodic)
        """
        
        def calc_chronicity(event_series):
            """Calculate for single location"""
            if not isinstance(event_series, np.ndarray):
                event_series = np.array(event_series)
            
            if event_series.sum() == 0:
                return 0.0
            
            # Find event durations
            # Label consecutive True values
            events_labeled = np.zeros_like(event_series, dtype=int)
            current_event = 0
            in_event = False
            
            for i, val in enumerate(event_series):
                if val > 0:
                    if not in_event:
                        current_event += 1
                        in_event = True
                    events_labeled[i] = current_event
                else:
                    in_event = False
            
            if current_event == 0:
                return 0.0
            
            # Calculate mean duration
            durations = []
            for event_id in range(1, current_event + 1):
                duration = np.sum(events_labeled == event_id)
                durations.append(duration)
            
            mean_duration = np.mean(durations)
            n_events = len(durations)
            
            # Chronicity = mean duration / event frequency
            # Normalize by total time
            chronicity = mean_duration / (len(event_series) / n_events)
            
            return chronicity
        
        chronicity = xr.apply_ufunc(
            calc_chronicity,
            events,
            input_core_dims=[['time']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        
        return chronicity
    
    
    def _calculate_incidental_to_chronic_year(
        self,
        events: xr.DataArray,
        chronic_threshold: float = 0.10
    ) -> xr.DataArray:
        """
        Calculate the first year when stressor exposure transitions from incidental
        (episodic) to chronic (persistent).

        For each grid cell: cumulative fraction of time under stress is computed
        year-by-year. The transition year is the first year when this cumulative
        exposure exceeds `chronic_threshold` (default 0.10 = 10% of time).

        Returns
        -------
        xr.DataArray
            Year of transition (int). 0 or NaN = never reached chronic threshold.
        """
        def incidental_to_chronic_year(event_series, years):
            """Get first year when cumulative exposure exceeds chronic threshold."""
            if not isinstance(event_series, np.ndarray):
                event_series = np.asarray(event_series)
            if not isinstance(years, np.ndarray):
                years = np.asarray(years)

            # Extract year for each timestep (handle cftime)
            yr_vals = np.array([
                int(getattr(y, 'year', y)) if hasattr(y, 'year') else int(y)
                for y in years
            ])

            unique_years = np.unique(yr_vals)
            if len(unique_years) == 0:
                return 0

            # Cumulative exposure: for each year, fraction of time under stress from start to that year
            for y in np.sort(unique_years):
                mask = yr_vals <= y
                if mask.sum() == 0:
                    continue
                cum_frac = event_series[mask].sum() / mask.sum()
                if cum_frac >= chronic_threshold:
                    if 2017 <= y <= 2060:
                        return int(y)
                    return 0
            return 0

        years = events.time.values
        trans_year = xr.apply_ufunc(
            incidental_to_chronic_year,
            events,
            input_core_dims=[['time']],
            kwargs={'years': years},
            vectorize=True,
            dask='parallelized',
            output_dtypes=[int],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        # Replace 0 with NaN for "never reached" (cleaner for plotting)
        trans_year = trans_year.where(trans_year > 0)
        return trans_year
    
    
    def _calculate_seasonal_coverage(self, events: xr.DataArray) -> xr.DataArray:
        """Calculate number of months per year with at least one event"""
        
        # Group by year and month
        monthly = events.groupby('time.year').apply(
            lambda x: x.groupby('time.month').any('time').sum('month')
        )
        
        # Mean over years
        mean_months = monthly.mean('year')
        
        return mean_months
    
    
    def calculate_rate_of_change(
        self,
        data_arrays: Dict[str, xr.DataArray],
        window_years: int = 10
    ) -> Dict[str, xr.DataArray]:
        """
        SUGGESTION 4: Rate of Change Analysis
        
        Analyze acceleration of environmental change beyond simple trends.
        Calculate rolling rates of change to identify periods of rapid vs.
        gradual intensification.
        
        Parameters
        ----------
        data_arrays : dict
            Dictionary of environmental variable DataArrays
        window_years : int
            Window for calculating rolling rate of change
            
        Returns
        -------
        dict
            Rate of change metrics and acceleration patterns
        """
        print("\n" + "="*80)
        print("SUGGESTION 4: RATE OF CHANGE ANALYSIS")
        print("="*80)
        
        results = {}
        
        for name, data in data_arrays.items():
            print(f"\nAnalyzing rate of change for {name}...")
            
            if 'time' not in data.dims:
                print(f"  Skipping {name}: no time dimension")
                continue
            
            # Calculate annual means
            annual = data.groupby('time.year').mean('time')
            
            # Ensure year coordinate is properly set (should be integer years)
            if 'year' not in annual.coords or annual.year.dtype != 'int32':
                # Extract year values from the groupby result
                year_values = annual.year.values
                # Ensure they are integers
                if not np.issubdtype(year_values.dtype, np.integer):
                    year_values = year_values.astype(int)
                annual = annual.assign_coords(year=year_values)
            
            # Overall linear trend
            trend = self._calculate_spatial_trend(annual, 'year')
            results[f"{name}_linear_trend"] = trend
            
            # Rolling rate of change
            if len(annual.year) >= window_years:
                rolling_rate = self._calculate_rolling_rate(annual, window_years)
                results[f"{name}_rolling_rate_{window_years}yr"] = rolling_rate

                # Acceleration (change in rate of change)
                # Ensure year coordinate is properly set for differentiate
                if 'year' not in rolling_rate.coords:
                    # Reassign year coordinate from annual
                    rolling_rate = rolling_rate.assign_coords(year=annual.year)
                
                # Rechunk to avoid small chunks for differentiate operation
                if hasattr(rolling_rate.data, 'chunks'):
                    rolling_rate_rechunked = rolling_rate.chunk({'year': -1})
                else:
                    rolling_rate_rechunked = rolling_rate
                
                # Calculate acceleration as derivative of rolling rate with respect to year
                # differentiate calculates d(rolling_rate)/d(year), which gives acceleration
                # Ensure year coordinate is evenly spaced (required for differentiate)
                year_values = rolling_rate_rechunked.year.values
                year_diff = np.diff(year_values)
                if not np.allclose(year_diff, year_diff[0], rtol=1e-6):
                    print(f"  ⚠ Warning: Year coordinate is not evenly spaced for {name}")
                    print(f"    Year differences: {year_diff[:5]}... (should be constant)")
                    # Interpolate to evenly spaced years if needed
                    year_min, year_max = int(year_values.min()), int(year_values.max())
                    year_even = np.arange(year_min, year_max + 1)
                    rolling_rate_rechunked = rolling_rate_rechunked.interp(year=year_even)
                
                acceleration = rolling_rate_rechunked.differentiate('year')
                
                # For temperature variables, acceleration should generally be positive in a warming world
                # If we get mostly negative values, there might be an issue with the calculation
                # Let's validate: if mean rolling rate is positive (warming) but acceleration is negative,
                # we should check if this is a real deceleration or a calculation error
                # Compute statistics - need to handle dask arrays properly
                mean_rolling_rate = float(rolling_rate.mean().compute() if hasattr(rolling_rate.data, 'chunks') else rolling_rate.mean())
                mean_accel = float(acceleration.mean().compute() if hasattr(acceleration.data, 'chunks') else acceleration.mean())
                # For median, dask requires computing first or converting to numpy
                if hasattr(acceleration.data, 'chunks'):
                    median_accel = float(np.nanmedian(acceleration.compute().values))
                    accel_min = float(acceleration.min().compute())
                    accel_max = float(acceleration.max().compute())
                else:
                    median_accel = float(acceleration.median())
                    accel_min = float(acceleration.min())
                    accel_max = float(acceleration.max())
                
                # Print diagnostic information
                print(f"  Acceleration statistics:")
                print(f"    Mean: {mean_accel:.6e} units/year²")
                print(f"    Median: {median_accel:.6e} units/year²")
                print(f"    Min: {accel_min:.6e}, Max: {accel_max:.6e}")
                
                # For temperature (tos_surface, thetao_bottom), if warming is occurring,
                # negative acceleration might indicate decelerating warming (still warming but slower)
                # However, in a warming world with increasing GHGs, we'd expect acceleration
                # If acceleration is strongly negative, log a warning
                if name in ['tos_surface', 'thetao_bottom']:
                    if mean_rolling_rate > 0:
                        if mean_accel < -1e-6:
                            print(f"  ⚠ Warning: {name} shows positive warming rate ({mean_rolling_rate:.6f} °C/year)")
                            print(f"    but negative acceleration ({mean_accel:.6e} °C/year²)")
                            print(f"    This suggests decelerating warming (still warming but slower)")
                            print(f"    If this is unexpected, check data quality and calculation method")
                        elif mean_accel > 1e-6:
                            print(f"  ✓ {name} shows accelerating warming (acceleration = {mean_accel:.6e} °C/year²)")
                    else:
                        print(f"  ⚠ Warning: {name} shows negative trend ({mean_rolling_rate:.6f} °C/year)")
                        print(f"    This is unexpected for temperature in a warming world")
                
                results[f"{name}_acceleration"] = acceleration
                
                # Identify rapid change periods (>90th percentile of rate)
                # Rechunk before quantile to avoid dask parallelization issues
                rolling_rate_for_quantile = rolling_rate.chunk({'year': -1}) if hasattr(rolling_rate.data, 'chunks') else rolling_rate
                threshold = rolling_rate_for_quantile.quantile(0.9, dim='year')
                # Drop quantile coordinate to avoid merge conflicts
                if 'quantile' in threshold.coords:
                    threshold = threshold.drop_vars('quantile')
                rapid_change = rolling_rate > threshold
                results[f"{name}_rapid_change_periods"] = rapid_change
                
                print(f"  Mean linear trend: {float(trend.mean()):.6f} per year")
                print(f"  Mean rolling rate: {float(rolling_rate.mean()):.6f} per year")
        
        # Cross-stressor rate comparisons
        print("\nCalculating synchronized rate changes...")
        if len(results) >= 2:
            rate_keys = [k for k in results.keys() if 'rolling_rate' in k]
            if len(rate_keys) >= 2:
                # Correlation of rates across stressors
                results['rate_synchronization'] = self._calculate_rate_correlation(results, rate_keys)
        
        return results
    
    
    def _calculate_spatial_trend(self, data: xr.DataArray, dim: str) -> xr.DataArray:
        """Calculate linear trend for each spatial location"""
        
        # Rechunk spatial dimensions to single chunks to avoid dask parallelization issues
        if hasattr(data.data, 'chunks'):
            rechunk_dict = {}
            if 'lat' in data.dims:
                rechunk_dict['lat'] = -1
            if 'lon' in data.dims:
                rechunk_dict['lon'] = -1
            if rechunk_dict:
                data = data.chunk(rechunk_dict)
        
        def calc_trend(values):
            if len(values) < 3:
                return np.nan
            x = np.arange(len(values))
            try:
                slope, _, _, _, _ = stats.linregress(x, values)
                return slope
            except:
                return np.nan
        
        trend = xr.apply_ufunc(
            calc_trend,
            data,
            input_core_dims=[[dim]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        
        return trend
    
    
    def _calculate_rolling_rate(self, data: xr.DataArray, window: int) -> xr.DataArray:
        """Calculate rolling rate of change"""

        def calc_rate_wrapper(values, axis):
            """Wrapper for rolling reduce that handles numpy arrays"""
            if not isinstance(values, np.ndarray):
                values = np.array(values)

            # Handle the case where axis is specified
            if axis is not None and len(values.shape) > 1:
                # Apply along the specified axis
                result_shape = list(values.shape)
                result_shape[axis] = 1
                result = np.full(result_shape, np.nan)

                # Calculate slope for the window
                x = np.arange(values.shape[axis])

                # Create indices for all other axes
                other_axes = [i for i in range(len(values.shape)) if i != axis]

                if len(other_axes) == 0:
                    # 1D case
                    if len(values) >= 2:
                        try:
                            slope, _, _, _, _ = stats.linregress(x, values)
                            result[0] = slope
                        except:
                            pass
                else:
                    # Multi-dimensional case - iterate over other dimensions
                    for idx in np.ndindex(*[values.shape[i] for i in other_axes]):
                        full_idx = list(idx)
                        full_idx.insert(axis, slice(None))
                        full_idx = tuple(full_idx)

                        window_data = values[full_idx]
                        if len(window_data) >= 2 and not np.all(np.isnan(window_data)):
                            try:
                                slope, _, _, _, _ = stats.linregress(x, window_data)
                                result_idx = list(idx)
                                result_idx.insert(axis, 0)
                                result[tuple(result_idx)] = slope
                            except:
                                pass

                return result.squeeze(axis=axis)
            else:
                # Simple 1D case
                if len(values) < 2:
                    return np.nan
                x = np.arange(len(values))
                try:
                    slope, _, _, _, _ = stats.linregress(x, values)
                    return slope
                except:
                    return np.nan

        # Rolling window calculation using simple reduce
        rolling_rate = data.rolling(year=window, center=True).reduce(calc_rate_wrapper)
        
        # Ensure year coordinate is preserved (rolling should preserve it, but verify)
        if 'year' not in rolling_rate.coords or len(rolling_rate.year) != len(data.year):
            # Reassign year coordinate from original data
            rolling_rate = rolling_rate.assign_coords(year=data.year.values)

        return rolling_rate
    
    
    def _calculate_rate_correlation(
        self, 
        results: Dict[str, xr.DataArray],
        rate_keys: List[str]
    ) -> xr.DataArray:
        """Calculate correlation between rates of change across stressors"""
        
        # Get first two rate fields
        rate1 = results[rate_keys[0]]
        rate2 = results[rate_keys[1]]
        
        # Calculate correlation along time dimension
        correlation = xr.corr(rate1, rate2, dim='year')
        
        return correlation
    
    
    def analyze_threshold_crossing(
        self,
        data_arrays: Dict[str, xr.DataArray],
        thresholds: Dict[str, Union[float, xr.DataArray]],
        baseline_period: Optional[Tuple[int, int]] = None
    ) -> Dict[str, xr.DataArray]:
        """
        SUGGESTION 5: Threshold Crossing Analysis

        Map timing of first exceedance and permanent crossing of
        ecologically relevant thresholds.

        Parameters
        ----------
        data_arrays : dict
            Dictionary of environmental variable DataArrays
        thresholds : dict
            Dictionary of threshold values for each variable. Can be:
            - float: Fixed threshold (e.g., {'ph_bottom': 7.75})
            - xr.DataArray: Day-of-year based threshold with 'dayofyear' dimension
        baseline_period : tuple, optional
            Baseline period for threshold definition

        Returns
        -------
        dict
            Threshold crossing metrics including first exceedance timing
        """
        print("\n" + "="*80)
        print("SUGGESTION 5: THRESHOLD CROSSING ANALYSIS")
        print("="*80)
        
        if baseline_period is None:
            baseline_period = self.baseline_period
        
        results = {}
        
        for name, data in data_arrays.items():
            # Check if threshold defined for this variable
            threshold_key = name
            if threshold_key not in thresholds:
                # Try to match partial keys
                matching = [k for k in thresholds.keys() if k in name]
                if len(matching) == 0:
                    continue
                threshold_key = matching[0]
            
            threshold = thresholds[threshold_key]

            if 'time' not in data.dims:
                continue

            # Check if threshold is DOY-based (DataArray) or scalar
            is_doy_threshold = isinstance(threshold, xr.DataArray) and 'dayofyear' in threshold.dims

            if is_doy_threshold:
                print(f"\nAnalyzing threshold crossing for {name} (DOY-based threshold)...")
                # Match threshold to data by day of year
                # Create a threshold array that matches the time dimension of data
                data_doy = data.time.dt.dayofyear
                threshold_matched = threshold.sel(dayofyear=data_doy)

                # Determine if this is a "less than" or "greater than" threshold
                if 'ph' in name.lower() or 'o2' in name.lower():
                    exceeds = data < threshold_matched
                    print(f"  Using 'less than' threshold (low values problematic)")
                else:
                    exceeds = data > threshold_matched
                    print(f"  Using 'greater than' threshold (high values problematic)")
            else:
                print(f"\nAnalyzing threshold crossing for {name} (threshold: {threshold})...")
                # Determine if this is a "less than" or "greater than" threshold
                # pH and O2: low values are bad (less than threshold)
                # Temperature: high values are bad (greater than threshold)
                if 'ph' in name.lower() or 'o2' in name.lower():
                    exceeds = data < threshold
                    print(f"  Using 'less than' threshold (low values problematic)")
                else:
                    exceeds = data > threshold
                    print(f"  Using 'greater than' threshold (high values problematic)")
            
            # First exceedance year
            first_exceedance = self._calculate_first_exceedance(exceeds)
            results[f"{name}_first_exceedance_year"] = first_exceedance
            
            # Permanent crossing (never returns below/above threshold)
            permanent_crossing = self._calculate_permanent_crossing(exceeds)
            results[f"{name}_permanent_crossing_year"] = permanent_crossing
            
            # Frequency of crossing during baseline
            baseline_mask = (
                (data.time.dt.year >= baseline_period[0]) &
                (data.time.dt.year <= baseline_period[1])
            )
            baseline_exceed = exceeds.where(baseline_mask, drop=True)
            baseline_frequency = baseline_exceed.sum('time') / baseline_exceed.sizes['time']
            results[f"{name}_baseline_exceed_frequency"] = baseline_frequency
            
            # Future frequency
            future_mask = data.time.dt.year > baseline_period[1]
            future_exceed = exceeds.where(future_mask, drop=True)
            if future_exceed.sizes['time'] > 0:
                future_frequency = future_exceed.sum('time') / future_exceed.sizes['time']
                results[f"{name}_future_exceed_frequency"] = future_frequency
                
                # Change in frequency
                frequency_change = future_frequency - baseline_frequency
                results[f"{name}_frequency_change"] = frequency_change
            
            # Calculate mean first exceedance year (skip NaN = never exceeded)
            mean_first = float(first_exceedance.mean(skipna=True))
            if not np.isnan(mean_first):
                print(f"  Mean first exceedance year: {mean_first:.1f}")
            
            # Percent of area with permanent crossing by end of period
            perm_crossed = int((permanent_crossing > 0).sum().compute())
            total_cells = permanent_crossing.size
            pct = (perm_crossed / total_cells) * 100
            print(f"  Permanent crossing by end: {pct:.1f}% of area")
        
        return results
    
    
    def _calculate_first_exceedance(self, exceeds: xr.DataArray) -> xr.DataArray:
        """Calculate year of first threshold exceedance.
        Returns NaN for never exceeded; invalid years (e.g. 2200) replaced with NaN.
        """
        def first_exceed_year(exceed_series, years):
            """Get first year where threshold exceeded"""
            if not isinstance(exceed_series, np.ndarray):
                exceed_series = np.asarray(exceed_series)
            exceed_indices = np.where(exceed_series)[0]
            if len(exceed_indices) == 0:
                return 0  # Never exceeded
            y = years[exceed_indices[0]]
            yr = int(getattr(y, 'year', y)) if hasattr(y, 'year') else int(y)
            if yr < 2015 or yr > 2070:
                return 0
            return yr

        years = exceeds.time.dt.year.values
        if hasattr(years, 'ravel'):
            years = np.asarray(years.ravel())

        first_year = xr.apply_ufunc(
            first_exceed_year,
            exceeds,
            input_core_dims=[['time']],
            kwargs={'years': years},
            vectorize=True,
            dask='parallelized',
            output_dtypes=[int],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        # Replace invalid years (e.g. cftime/dtype artifacts) with NaN (never exceeded)
        valid = (first_year >= 2017) & (first_year <= 2060)
        first_year = first_year.where(valid)
        return first_year
    
    
    def _calculate_permanent_crossing(self, exceeds: xr.DataArray) -> xr.DataArray:
        """Calculate year when threshold is permanently crossed"""
        
        def permanent_year(exceed_series, years):
            """Get year when threshold permanently exceeded"""
            if not isinstance(exceed_series, np.ndarray):
                exceed_series = np.array(exceed_series)
            
            # Find last False in series
            non_exceed_indices = np.where(~exceed_series)[0]
            
            if len(non_exceed_indices) == 0:
                # Always exceeded - return first year
                return years[0]
            
            last_ok = non_exceed_indices[-1]
            
            # Check if permanently exceeded after this point
            if last_ok == len(exceed_series) - 1:
                # Never permanently crossed
                return 0
            
            # Year after last OK value
            return years[last_ok + 1]
        
        years = exceeds.time.dt.year.values
        
        permanent = xr.apply_ufunc(
            permanent_year,
            exceeds,
            input_core_dims=[['time']],
            kwargs={'years': years},
            vectorize=True,
            dask='parallelized',
            output_dtypes=[int],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        
        return permanent
    
    
    def assess_refugia_stability(
        self,
        extremes: Dict[str, xr.Dataset],
        intensity_threshold: float = 0.5
    ) -> Dict[str, xr.DataArray]:
        """
        SUGGESTION 6: Spatial Refugia Stability
        
        Assess temporal stability of refugia (low stressor areas).
        Calculate "half-life" of refugia protection.
        
        Parameters
        ----------
        extremes : dict
            Dictionary of extreme event datasets
        intensity_threshold : float
            Intensity below which area is considered refugia (default: 0.5)
            
        Returns
        -------
        dict
            Refugia stability metrics including half-life and deterioration rates
        """
        print("\n" + "="*80)
        print("SUGGESTION 6: SPATIAL REFUGIA STABILITY")
        print("="*80)
        
        results = {}
        
        for name, data in extremes.items():
            print(f"\nAnalyzing refugia stability for {name}...")
            
            if 'intensity' not in data and 'mean_intensity' not in data:
                continue
            
            intensity_var = 'mean_intensity' if 'mean_intensity' in data else 'intensity'
            intensity = data[intensity_var]
            
            if 'time' not in intensity.dims:
                continue
            
            # Annual mean intensity
            annual_intensity = intensity.groupby('time.year').mean('time')
            
            # Define refugia: areas where intensity < threshold
            is_refugia = annual_intensity < intensity_threshold
            
            # Calculate refugia persistence
            refugia_years = is_refugia.sum('year')
            results[f"{name}_refugia_years"] = refugia_years
            
            # Refugia "half-life": year when refugia lost for half the time series
            total_years = len(annual_intensity.year)
            half_life = self._calculate_refugia_halflife(is_refugia)
            results[f"{name}_refugia_halflife_year"] = half_life
            
            # Refugia deterioration rate
            deterioration = self._calculate_refugia_deterioration(is_refugia)
            results[f"{name}_refugia_deterioration_rate"] = deterioration
            
            # Identify persistent refugia (refugia for >80% of time)
            persistent_threshold = 0.8 * total_years
            persistent_refugia = refugia_years > persistent_threshold
            results[f"{name}_persistent_refugia"] = persistent_refugia
            
            # Identify lost refugia (was refugia at start, lost by end)
            early_refugia = is_refugia.isel(year=slice(0, min(5, total_years//4))).all('year')
            late_refugia = is_refugia.isel(year=slice(-min(5, total_years//4), None)).all('year')
            lost_refugia = early_refugia & ~late_refugia
            results[f"{name}_lost_refugia"] = lost_refugia
            
            # Statistics
            total_cells = refugia_years.size
            persistent_count = persistent_refugia.sum().item()
            lost_count = lost_refugia.sum().item()
            
            print(f"  Persistent refugia: {(persistent_count/total_cells)*100:.1f}% of area")
            print(f"  Lost refugia: {(lost_count/total_cells)*100:.1f}% of area")
            
            mean_halflife = float(half_life.where(half_life > 0).mean())
            if not np.isnan(mean_halflife):
                print(f"  Mean refugia half-life: {mean_halflife:.1f} years")
        
        return results
    
    
    def _calculate_refugia_halflife(self, is_refugia: xr.DataArray) -> xr.DataArray:
        """Calculate year when refugia status is lost for >50% of remaining time"""
        
        def calc_halflife(refugia_series, years):
            """Calculate half-life for a single location"""
            if not isinstance(refugia_series, np.ndarray):
                refugia_series = np.array(refugia_series)
            
            # Find first year where refugia lost for majority of remaining time
            n_years = len(refugia_series)
            
            for i in range(n_years):
                remaining = refugia_series[i:]
                if np.sum(remaining) < len(remaining) * 0.5:
                    return years[i]
            
            return 0  # Never lost refugia status
        
        years = is_refugia.year.values
        
        halflife = xr.apply_ufunc(
            calc_halflife,
            is_refugia,
            input_core_dims=[['year']],
            kwargs={'years': years},
            vectorize=True,
            dask='parallelized',
            output_dtypes=[int],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        
        return halflife
    
    
    def _calculate_refugia_deterioration(self, is_refugia: xr.DataArray) -> xr.DataArray:
        """Calculate rate of refugia loss (slope of cumulative loss)"""
        
        def calc_deterioration(refugia_series):
            """Calculate deterioration rate"""
            if not isinstance(refugia_series, np.ndarray):
                refugia_series = np.array(refugia_series)
            
            # Convert to refugia loss (1 = refugia, 0 = not)
            # Calculate cumulative loss
            cumulative_refugia = np.cumsum(refugia_series)
            years = np.arange(len(refugia_series))
            
            try:
                # Negative slope = losing refugia over time
                slope, _, _, _, _ = stats.linregress(years, cumulative_refugia)
                return -slope  # Return as positive for deterioration
            except:
                return np.nan
        
        deterioration = xr.apply_ufunc(
            calc_deterioration,
            is_refugia,
            input_core_dims=[['year']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        
        return deterioration
    
    
    def analyze_cross_shelf_gradients(
        self,
        data_arrays: Dict[str, xr.DataArray],
        coast_lon: float = -124.5,
        offshore_extent: float = 2.5,
        n_bins: int = 10
    ) -> Dict[str, xr.DataArray]:
        """
        SUGGESTION 7: Cross-Shelf Gradient Analysis

        Quantify how stressor intensity varies from nearshore to offshore
        and analyze whether this gradient is steepening or flattening.

        Parameters
        ----------
        data_arrays : dict
            Dictionary of environmental variable DataArrays
        coast_lon : float
            Approximate longitude of coast (default: -124.5 for Washington)
        offshore_extent : float
            Degrees longitude to extend offshore (default: 2.5)
        n_bins : int
            Number of bins for gradient calculation

        Returns
        -------
        dict
            Cross-shelf gradient metrics and temporal changes
        """
        print("\n" + "="*80)
        print("SUGGESTION 7: CROSS-SHELF GRADIENT ANALYSIS")
        print("="*80)

        results = {}

        for name, data in data_arrays.items():
            print(f"\nAnalyzing cross-shelf gradient for {name}...")

            if 'lon' not in data.dims and 'longitude' not in data.dims:
                print(f"  ⚠ Skipping: No longitude dimension found")
                continue

            lon_dim = 'lon' if 'lon' in data.dims else 'longitude'

            # Get actual data longitude range
            lon_min = float(data[lon_dim].min())
            lon_max = float(data[lon_dim].max())
            print(f"  Data longitude range: [{lon_min:.2f}, {lon_max:.2f}]")

            # Use actual data range instead of fixed coast_lon
            # From nearshore (max lon, closest to coast) to offshore (min lon, farther west)
            lon_bins = np.linspace(lon_max, lon_min, n_bins + 1)
            bin_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

            print(f"  Using {n_bins} bins from {lon_max:.2f} (nearshore) to {lon_min:.2f} (offshore)")

            # Calculate mean for each cross-shelf bin
            binned = []
            valid_bins = 0
            for i in range(len(lon_bins) - 1):
                # Create mask for this bin (from larger to smaller lon values)
                mask = (data[lon_dim] >= lon_bins[i+1]) & (data[lon_dim] <= lon_bins[i])
                n_points = int(mask.sum())

                if n_points > 0:
                    # Use skipna=True to handle NaN values properly
                    binned_data = data.where(mask).mean(dim=[lon_dim, 'lat'], skipna=True)
                    binned.append(binned_data)
                    valid_bins += 1
                else:
                    # Create NaN-filled array with correct dimensions
                    binned_data = data.isel({lon_dim: 0, 'lat': 0}).copy()
                    binned_data.values = np.nan
                    binned.append(binned_data)

            print(f"  Valid bins with data: {valid_bins}/{n_bins}")

            if valid_bins == 0:
                print(f"  ⚠ Warning: No valid data in any bin, skipping gradient analysis")
                continue

            # Stack into new dimension
            shelf_profile = xr.concat(binned, dim='distance')
            shelf_profile['distance'] = np.arange(n_bins)  # 0 = nearshore, n-1 = offshore

            # Drop quantile coordinate if present to avoid merge conflicts
            if 'quantile' in shelf_profile.coords:
                shelf_profile = shelf_profile.drop_vars('quantile')

            results[f"{name}_shelf_profile"] = shelf_profile

            # Calculate gradient strength (nearshore - offshore difference)
            if 'time' in shelf_profile.dims:
                nearshore = shelf_profile.isel(distance=0)
                offshore = shelf_profile.isel(distance=-1)
                gradient_strength = offshore - nearshore

                # Drop quantile coordinate if present
                if 'quantile' in gradient_strength.coords:
                    gradient_strength = gradient_strength.drop_vars('quantile')

                results[f"{name}_gradient_strength"] = gradient_strength

                # Gradient trend over time
                annual_gradient = gradient_strength.groupby('time.year').mean('time')
                gradient_trend = self._calculate_spatial_trend(annual_gradient, 'year')

                # Drop quantile coordinate if present
                if 'quantile' in gradient_trend.coords:
                    gradient_trend = gradient_trend.drop_vars('quantile')

                results[f"{name}_gradient_trend"] = gradient_trend

                mean_grad = float(gradient_strength.mean())
                if not np.isnan(mean_grad):
                    print(f"  Mean gradient strength: {mean_grad:.4f}")
                    print(f"  Gradient trend: {float(gradient_trend):.6f} per year")
                else:
                    print(f"  ⚠ Warning: Gradient strength is NaN (all bin means are NaN)")

            # Calculate gradient steepness (slope across shelf)
            gradient_slope = self._calculate_shelf_gradient_slope(shelf_profile)

            # Drop quantile coordinate if present
            if 'quantile' in gradient_slope.coords:
                gradient_slope = gradient_slope.drop_vars('quantile')

            results[f"{name}_gradient_slope"] = gradient_slope

        return results
    
    
    def _calculate_shelf_gradient_slope(self, shelf_profile: xr.DataArray) -> xr.DataArray:
        """Calculate slope of cross-shelf profile"""
        
        def calc_slope(profile):
            """Calculate slope for a single time point"""
            if len(profile) < 2:
                return np.nan
            
            x = np.arange(len(profile))
            try:
                slope, _, _, _, _ = stats.linregress(x, profile)
                return slope
            except:
                return np.nan
        
        if 'time' in shelf_profile.dims:
            slope = xr.apply_ufunc(
                calc_slope,
                shelf_profile,
                input_core_dims=[['distance']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'allow_rechunk': True}
            )
        else:
            slope = calc_slope(shelf_profile.values)
        
        return slope
    
    
    def calculate_stressor_interaction_index(
        self,
        extremes: Dict[str, xr.Dataset],
        weights: Optional[Dict[str, float]] = None,
        method: str = 'additive'
    ) -> Dict[str, xr.DataArray]:
        """
        SUGGESTION 10: Stressor Interaction Index
        
        Develop composite index showing combined stressor exposure.
        Can use additive or multiplicative approach with optional weighting.
        
        Parameters
        ----------
        extremes : dict
            Dictionary of extreme event datasets
        weights : dict, optional
            Weights for each stressor (default: equal weights)
        method : str
            'additive' or 'multiplicative' (default: 'additive')
            
        Returns
        -------
        dict
            Interaction indices and standardized exposure scores
        """
        print("\n" + "="*80)
        print(f"SUGGESTION 10: STRESSOR INTERACTION INDEX ({method.upper()})")
        print("="*80)
        
        results = {}
        
        # Extract intensity measures
        intensities = {}
        for name, data in extremes.items():
            if 'mean_intensity' in data:
                intensities[name] = data['mean_intensity']
            elif 'intensity' in data:
                intensities[name] = data['intensity'].mean('time') if 'time' in data['intensity'].dims else data['intensity']
        
        if len(intensities) == 0:
            print("Warning: No intensity data available")
            return results
        
        # Set default weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in intensities.keys()}
        
        print(f"\nWeights: {weights}")
        
        # Normalize intensities to 0-1 scale
        normalized = {}
        for name, intensity in intensities.items():
            max_val = float(intensity.max())
            if max_val > 0:
                normalized[name] = intensity / max_val
            else:
                normalized[name] = intensity
        
        # Calculate interaction index
        if method == 'additive':
            # Weighted sum
            index = xr.zeros_like(list(normalized.values())[0])
            total_weight = sum(weights.values())
            
            for name, norm_intensity in normalized.items():
                weight = weights.get(name, 1.0)
                index = index + (norm_intensity * weight)
            
            index = index / total_weight
            results['stressor_interaction_index_additive'] = index
            
        elif method == 'multiplicative':
            # Weighted product
            index = xr.ones_like(list(normalized.values())[0])
            
            for name, norm_intensity in normalized.items():
                weight = weights.get(name, 1.0)
                # Add small value to avoid issues with zeros
                weighted = (norm_intensity + 0.01) ** weight
                index = index * weighted
            
            # Normalize
            index = index / float(index.max())
            results['stressor_interaction_index_multiplicative'] = index
        
        # Drop quantile coordinate if present
        if 'quantile' in index.coords:
            index = index.drop_vars('quantile')

        # Calculate exposure score (0-100 scale)
        exposure_score = index * 100

        # Drop quantile coordinate if present
        if 'quantile' in exposure_score.coords:
            exposure_score = exposure_score.drop_vars('quantile')

        results['exposure_score'] = exposure_score

        # Classify exposure levels
        low_exposure = exposure_score < 25
        moderate_exposure = (exposure_score >= 25) & (exposure_score < 50)
        high_exposure = (exposure_score >= 50) & (exposure_score < 75)
        extreme_exposure = exposure_score >= 75

        results['low_exposure_mask'] = low_exposure
        results['moderate_exposure_mask'] = moderate_exposure
        results['high_exposure_mask'] = high_exposure
        results['extreme_exposure_mask'] = extreme_exposure
        
        # Summary statistics
        print(f"\nExposure Score Statistics:")
        print(f"  Mean: {float(exposure_score.mean()):.1f}")
        print(f"  Low (<25): {float(low_exposure.sum() / low_exposure.size * 100):.1f}%")
        print(f"  Moderate (25-50): {float(moderate_exposure.sum() / moderate_exposure.size * 100):.1f}%")
        print(f"  High (50-75): {float(high_exposure.sum() / high_exposure.size * 100):.1f}%")
        print(f"  Extreme (>75): {float(extreme_exposure.sum() / extreme_exposure.size * 100):.1f}%")
        
        return results
    
    
    def _is_daily_data(self, data: xr.DataArray) -> bool:
        """Check if data is daily resolution"""
        if 'time' not in data.dims or len(data.time) < 2:
            return False
        
        # Calculate median time difference
        time_diff = np.diff(data.time.values.astype('datetime64[D]').astype(int))
        median_diff = np.median(time_diff)
        
        return median_diff <= 1  # 1 day or less


# End of AdvancedStressorMetrics class
