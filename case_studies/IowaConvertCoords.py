import glob
import os
import pickle
import pandas
import numpy
from pyproj import Proj, transform


root = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Re_Analysis/IowaRiver/masks'
name = 'IowaDownstream2'
path = glob.glob(os.path.join(root, name, 'bar_migration', '*csv'))[0]
df = pandas.read_csv(path)

inProj = Proj(init='epsg:32615')
outProj = Proj(init='epsg:4326')
lon, lat= transform(
    inProj,
    outProj,
    df['easting'],
    df['northing'],
)
df['lat'] = lat 
df['lon'] = lon 

df.to_csv(path)
