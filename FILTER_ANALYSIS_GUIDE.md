# Butterworth Filter Effects Analysis Guide

This guide explains how to use the new filter analysis tools to visualize the effects of the Butterworth filter on your accelerometer data.

## Overview

The actuator modeling pipeline applies an optional Butterworth low-pass filter to the tangential acceleration signal before calculating the target torque (`tau_measured`). This filtering helps remove high-frequency noise that may not represent actual actuator dynamics.

Two new tools have been added to help you analyze and understand the filter effects:

1. **`scripts/analyze_filter_effects.py`** - Main analysis tool
2. **`scripts/demo_filter_analysis.py`** - Demo with synthetic data

## Quick Start

### 1. Configure Filter Parameters

First, set up your filter parameters in `configs/data/default.yaml`:

```yaml
# Filter Parameters for ActuatorDataset
filter_cutoff_freq_hz: 50.0  # Cutoff frequency in Hz. Set to null to disable filtering
filter_order: 4               # Order of the Butterworth filter
```

### 2. Run Filter Analysis

```bash
# Use default filter settings from config
python scripts/analyze_filter_effects.py

# Override filter cutoff frequency
python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=30.0

# Change filter order
python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=50.0 data.filter_order=6

# Use a different data directory
python scripts/analyze_filter_effects.py data.data_base_dir="path/to/your/data"

# Disable filtering to see unfiltered data only
python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=null
```

### 3. View Results

The tool generates comprehensive plots in the `filter_analysis_plots/` directory, showing:

- **Before/after timeseries comparisons**
- **Power spectral density (PSD) plots**
- **Difference plots showing what the filter removes**
- **Quantitative metrics (RMS differences)**

## Understanding the Analysis Plots

Each generated plot contains 6 subplots arranged in a 2×3 grid:

### Top Row: Accelerometer Data Analysis
1. **Timeseries Comparison**: Raw vs. filtered accelerometer signals
2. **PSD Comparison**: Frequency content before and after filtering  
3. **Difference Plot**: What the filter removes from the signal

### Bottom Row: Derived Torque Analysis
4. **Torque Timeseries**: Unfiltered vs. filtered torque calculations
5. **Torque PSD**: Frequency content of the derived torque signals
6. **Torque Difference**: Impact of filtering on the final torque values

### Key Visual Elements

- **Blue lines**: Unfiltered data
- **Red lines**: Filtered data  
- **Green lines**: Difference (unfiltered - filtered)
- **Red dashed vertical line**: Filter cutoff frequency on PSD plots
- **Text boxes**: RMS difference values

## Choosing Filter Parameters

### Cutoff Frequency Selection

The cutoff frequency should be chosen based on:

1. **Signal content**: Examine the PSD plots to identify meaningful vs. noise frequencies
2. **Actuator bandwidth**: Consider the physical response characteristics of your actuator
3. **Sampling rate**: Ensure cutoff is well below Nyquist frequency (fs/2)

**General guidelines:**
- **Conservative**: Start with cutoff = fs/5 to fs/4
- **Aggressive**: Use cutoff = fs/10 to fs/8 for heavy noise reduction
- **Data-driven**: Use PSD analysis to identify noise floor

### Filter Order Selection

Higher order = steeper roll-off but potential artifacts:

- **Order 2-4**: Good balance for most applications
- **Order 6-8**: Steeper roll-off, use carefully
- **Order >8**: Generally not recommended

### Example Analysis Workflow

1. **Start with no filtering** (`filter_cutoff_freq_hz: null`)
   ```bash
   python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=null
   ```
   
2. **Examine raw data PSD** to identify noise frequencies

3. **Test different cutoff frequencies**:
   ```bash
   python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=20.0
   python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=40.0
   python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=60.0
   ```

4. **Compare results** and choose based on:
   - Noise reduction effectiveness
   - Preservation of signal content
   - RMS difference values

## Demo with Synthetic Data

Run the demonstration to see the filter in action:

```bash
python scripts/demo_filter_analysis.py
```

The demo:
- Generates synthetic actuator data with artificial high-frequency noise
- Applies a 20 Hz filter to remove noise at 25, 35, and 60 Hz
- Shows clear before/after comparison

## Integration with Training Pipeline

The filter analysis uses the **exact same filtering implementation** as the training pipeline:

- Same `scipy.signal.filtfilt` function (zero-phase filtering)
- Same parameter interpretation
- Same data processing workflow

This ensures that what you see in the analysis is exactly what happens during model training.

## Troubleshooting

### Common Issues

1. **"No CSV files found"**
   - Check your `data_base_dir` path
   - Verify directory structure matches config
   - Ensure CSV files exist in inertia group folders

2. **"Cannot determine sampling frequency"**
   - Check Time_ms column in your CSV files
   - Ensure data has consistent time intervals
   - Verify Time_ms values are reasonable

3. **"Filter cutoff >= Nyquist frequency"**
   - Reduce filter cutoff frequency
   - Check your data's sampling rate
   - Use `filter_cutoff_freq_hz < 0.5 * sampling_rate`

4. **"Not enough data points to filter"**
   - Increase data length
   - Reduce filter order
   - Check for valid data in CSV files

### Performance Tips

- **Limit files processed**: Analysis processes first 3 files per inertia group by default
- **Use lower sampling rates**: For initial exploration, downsample if needed
- **Parallel processing**: Set `num_workers > 0` if you have many files

## Advanced Usage

### Custom Analysis

You can import and use the analysis functions directly:

```python
from scripts.analyze_filter_effects import load_dataset_pair, analyze_accelerometer_signal

# Load your data with different filter settings
dataset_unfiltered, dataset_filtered = load_dataset_pair(
    csv_file="your_data.csv",
    inertia=0.02,
    radius_accel=0.3,
    gyro_axis='Gyro_Z',
    accel_axis='Acc_Y',
    filter_cutoff_freq_hz=25.0,
    filter_order=6
)

# Run analysis
analyze_accelerometer_signal(
    dataset_unfiltered=dataset_unfiltered,
    dataset_filtered=dataset_filtered,
    accel_axis='Acc_Y',
    output_dir=Path("custom_output"),
    file_label="my_analysis"
)
```

### Integration with Existing Analysis

The filter analysis complements the existing `scripts/explore_data.py` tool:

- **explore_data.py**: Overall data exploration and basic statistics
- **analyze_filter_effects.py**: Specific focus on filtering effects and frequency analysis

Use both tools together for comprehensive data understanding.

## Output Files

The analysis generates:

- **Individual plots**: `{group_id}_{filename}_filter_analysis.png`
- **Summary file**: `analysis_summary.txt`
- **Directory structure**: `filter_analysis_plots/`

## Next Steps

After running the filter analysis:

1. **Choose optimal filter parameters** based on the analysis
2. **Update your config** with the chosen parameters
3. **Run training** with the optimized filter settings
4. **Compare model performance** with different filter configurations

The goal is to find the filter parameters that provide the best balance between noise reduction and signal preservation for your specific actuator and application.