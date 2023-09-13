import os
import glob
import pickle
import pandas 
import numpy as np
from scipy.signal import savgol_filter
import cline_analysis as ca
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from scipy import spatial
from scipy.signal import medfilt
import functools
from scipy.optimize import bisect
from scipy import stats


class pickData(object):
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax):
        self.ax = ax
        self.events = []
        self.points = []
        self.pairs = []
        self.point_coords = []

    def clear(self, event):
        # Clear the most recent box of pointd
        self.events = []
        self.points = []
        self.X0 = None

        # Remove all plotted picked points
        for p in self.points:
            if p:
                p.remove()

        # Remove most recent rectangle
        self.points = []
        print('Cleared')

    def done(self, event):
        # All done picking points
        plt.close('all')
        print('All Done')

    def __call__(self, event):
        # Call the events
        self.event = event

        if not event.dblclick:
            return 0

        self.x, self.y = event.xdata, event.ydata 
        self.events.append((self.x, self.y))
        self.events = list(set(self.events))

        if self.x is not None:
            # Plot where the picked point is
            self.points.append(self.ax.scatter(self.x, self.y))
            self.point_coords.append([self.x, self.y])
            event.canvas.draw()

        if len(self.events) == 2:
            print(self.x, self.y)
            self.pairs.append(self.point_coords)
            self.events = []
            self.points = []
            self.point_coords = []


def pick_cutoffs(x, y, xn, yn):
    fig, ax = plt.subplots(1, 1)
    im = ax.plot(x, y, label='1995')
    im2 = ax.plot(xn, yn, label='2015')
    ax.legend()
    t = plt.gca()
    PD = pickData(t)

    axclear = plt.axes([0.0, 0.0, 0.1, 0.1])
    bclear = Button(plt.gca(), 'Clear')
    bclear.on_clicked(PD.clear)

    axdone = plt.axes([0.2, 0.0, 0.1, 0.1])
    bdone = Button(plt.gca(), 'Done')
    bdone.on_clicked(PD.done)

    fig.canvas.mpl_connect('button_press_event', PD)
    # im.set_picker(5) # Tolerance in points
    plt.show()

    cutoff_pairs = PD.pairs
    if not len(cutoff_pairs):
        return np.array([])
    tree = spatial.KDTree(np.array([x, y]).T)
    dist, i = tree.query(cutoff_pairs)

    return i


def get_migration(df_1995, df_2015):
    years = 2015 - 1995
    x = df_1995['easting'].values 
    y = df_1995['northing'].values 

    xn = df_2015['easting'].values 
    yn = df_2015['northing'].values 
    i = pick_cutoffs(x, y, xn, yn)
    # i = []

    migr_rate, migr_sign, p, q = ca.get_migr_rate(
        x, y, xn, yn, years, 0
    )
    migr_rate = medfilt(
        savgol_filter(migr_rate,11,3),
        kernel_size=5
    )
    curv,s = ca.compute_curvature(x,y)
    curv = medfilt(
        savgol_filter(curv, 71, 3), 
        kernel_size=5
    )

    df_1995['Mr_myr'] = migr_rate
    df_1995['curv'] = curv 
    df_1995['cutoff'] = 0
    for c in i:
        df_1995.at[c[0]:c[1], 'Mr_myr'] = None
        df_1995.at[c[0]:c[1], 'cutoff'] = 1

    return df_1995, curv


def get_bar_migration(df_1995, curv):
    # Get Bar-scale rates
    asign = np.sign(curv)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    sign = np.where(signchange)[0]
    bar_mr = []
    data = {
        'easting': [],
        'northing': [],
        'B_median_m': [],
        'B_mean_m': [],
        'B_std_m': [],
        'B_sem_m': [],
        'Mr_median_my': [],
        'Mr_std_my': [],
        'Mr_sem_my': [],
        'curv_mean': [],
        'curv_std': [],
        'curv_sem': [],
        'length_m': [],
    }
    for i0 , i1 in zip(sign, sign[1:]):
        bar = df_1995.iloc[i0:i1]
        if len(bar) <= 5:
            continue

        length = np.linalg.norm(
            bar.iloc[0][['easting', 'northing']]
            - bar.iloc[-1][['easting', 'northing']]
        )
        data['easting'].append(bar.mean()['easting'])
        data['northing'].append(bar.mean()['northing'])
        data['B_median_m'].append(bar['width_m'].median())
        data['B_mean_m'].append(bar['width_m'].abs().mean())
        data['B_std_m'].append(bar['width_m'].std())
        data['B_sem_m'].append(bar['width_m'].sem())
        data['Mr_median_my'].append(bar['Mr_myr'].abs().median())
        data['Mr_std_my'].append(bar['Mr_myr'].abs().std())
        data['Mr_sem_my'].append(bar['Mr_myr'].abs().sem())
        data['curv_mean'].append(bar['curv'].abs().mean())
        data['curv_std'].append(bar['curv'].abs().std())
        data['curv_sem'].append(bar['curv'].abs().sem())
        data['length_m'].append(length)

    bar_df = pandas.DataFrame(data=data)
    bar_df = bar_df.dropna(
        axis=0, 
        subset=['Mr_median_my', 'B_median_m']
    )

    return bar_df


path_1995 = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Re_Analysis/RedRiver/1995/width_dfs/width_dfs_1995.pickle'

with open(path_1995, 'rb') as handle:
    dfs_1995 = pickle.load(handle)

path_2015 = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Re_Analysis/RedRiver/2015/width_dfs/width_dfs_2015.pickle'

with open(path_2015, 'rb') as handle:
    dfs_2015 = pickle.load(handle)


keys_1995 = sorted(list(dfs_1995.keys()))
keys_2015 = sorted(list(dfs_2015.keys()))

root = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Re_Analysis/RedRiver/migration'
migration_dfs = {}
bar_dfs = {}
for key_1995, key_2015 in zip(keys_1995, keys_2015):
    i = int(key_1995.split('/')[-1].split('_')[-2])
    df_1995 = pandas.DataFrame(
        dfs_1995[key_1995], 
        columns=['rowi', 'coli', 'x', 'y', 'width', 'easting', 'northing']
    )
    df_1995['width_m'] = df_1995['width'] * 30

    df_2015 = pandas.DataFrame(
        dfs_2015[key_2015], 
        columns=['rowi', 'coli', 'x', 'y', 'width', 'easting', 'northing']
    )
    df_2015['width_m'] = df_2015['width'] * 30

    df_1995, curv = get_migration(df_1995, df_2015)
    bar_1995 = get_bar_migration(df_1995, curv)

    outpath = os.path.join(
        root,
        'points',
        'migration_{}_1995_2015.csv'.format(i)
    )
    df_1995.to_csv(outpath)

    outpath = os.path.join(
        root,
        'bar',
        'bar_migration_{}_1995_2015.csv'.format(i)
    )
    bar_1995.to_csv(outpath)

    migration_dfs[i] = df_1995
    bar_dfs[i] = bar_1995

full_migration = pandas.DataFrame()
for i, df in migration_dfs.items():
    df['i'] = i
    full_migration = pandas.concat([full_migration, df])

root = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Re_Analysis/RedRiver'
out_path = os.path.join(root, 'RedRiver_Point_Migration.csv')
full_migration.to_csv(out_path)

full_bar_migration = pandas.DataFrame()
for i, df in bar_dfs.items():
    df['i'] = i
    full_bar_migration = pandas.concat([full_bar_migration, df])

root = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Re_Analysis/RedRiver'
out_path = os.path.join(root, 'RedRiver_Bar_Migration.csv')
full_bar_migration.to_csv(out_path)
