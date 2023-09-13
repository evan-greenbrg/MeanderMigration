# MeanderMigration
Scripts accompanying "The Pace of Global River Meandering Set by Fluvial Sediment Supply "

## Codes:

**GetWBMdata_new_wbm.py**: 
Description: Will pull out annual averages from the WBM data for a list of coordinates.
Usage: This will require you to reach out to the author of WBMsed for the netCDF files with the WBM model output.

**cline_analysis.py**:
Sourced from: https://github.com/zsylvester/curvaturepy
Sylvester et al., (2019), "High curvatures drive river meandering".

Note: The dpcore package is derived from: https://github.com/dpwe/dp_python. This specific package is poorly supported now with the last update 9 years ago (as of 2023).
      It only works on Python 2, which can be troublesome. The dp_python package is included in its entirety in this repository.

**convert_to_csv.py**:
Description: A utility script to load a .pkl object generate from the make_centerline_meandering.py script. 
             The pickle object is a simple class so you unfortunately have to load the entire routine before you can load the object.
Usage: This can be used to open the .pkl objects found in the Dryad data repository

**get_migration.py**:
Description: Measures the migration rate for centerline coordinates between
             two timesteps.
Usage: Requires the cline_analysis, which itself uses the dpcore package. AGAIN NOTE: This HAS TO run with Python 2. 
       The cline_analysis scripts require the dp_python package, which is included in its entirety in this repository.
       Follow the dp_python installation instructions before running this code. See https://github.com/dpwe/dp_python for run through on how to install the package.

**make_centerline_meandering.py**:
Description: Generate a clean centerline vector object with measurements of width
             from a binary channel mask.
Usage: Could run in command line after swapping out variable definitions. I 
       commonly run in a side-by-side IPython terminal.


