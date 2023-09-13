import pickle
import pandas
import numpy
from pyproj import Proj, transform


path_1995 = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Re_Analysis/RedRiver/1995/width_dfs/width_dfs_1995.pickle'

with open(path_1995, 'rb') as handle:
    dfs_1995 = pickle.load(handle)

path_2015 = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Re_Analysis/RedRiver/2015/width_dfs/width_dfs_2015.pickle'

with open(path_2015, 'rb') as handle:
    dfs_2015 = pickle.load(handle)

keys_1995 = sorted(list(dfs_1995.keys()))
for key_1995 in keys_1995:
    df_1995 = pandas.DataFrame(
        dfs_1995[key_1995], 
        columns=['rowi', 'coli', 'x', 'y', 'width']
    )

    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:26915')
    easting, northing = transform(
        inProj,
        outProj,
        df_1995['x'],
        df_1995['y'],
    )
    df_1995['easting'] = easting
    df_1995['northing'] = northing 

    dfs_1995[key_1995] = df_1995.to_numpy()

with open(path_1995, 'wb') as handle:
    pickle.dump(dfs_1995, handle, protocol=2)

keys_2015 = sorted(list(dfs_2015.keys()))
for key_2015 in keys_2015:

    df_2015 = pandas.DataFrame(
        dfs_2015[key_2015], 
        columns=['rowi', 'coli', 'x', 'y', 'width']
    )

    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:26915')
    easting, northing = transform(
        inProj,
        outProj,
        df_2015['x'],
        df_2015['y'],
    )
    df_2015['easting'] = easting
    df_2015['northing'] = northing 

    dfs_2015[key_2015] = df_2015.to_numpy()

with open(path_2015, 'wb') as handle:
    pickle.dump(dfs_2015, handle, protocol=2)
