"""
Description: Measures the migration rate for centerline coordinates between
             two timesteps.
Usage: Requires the cline_analysis, which itself uses the dpcore package. NOTE: This HAS TO run with Python 2. 
       The cline_analysis scripts require the dp_python package, which is included in its entirety in this repository.
       Follow the dp_python installation instructions before running this code. 
"""
import pandas 
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
import cline_analysis as ca
import pandas as pd
import datetime
import os
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
    im = ax.plot(x, y, label='1991')
    im2 = ax.plot(xn, yn, label='2013')
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


river_root = '/Rio_Itui'
river = 'Rio_Itui'
root = os.path.join(river_root, 'centerline_csv')

y1 = 1990
year1 = pandas.read_csv(os.path.join(
    root, '{}_{}_centerline.csv'.format(river, y1)
))

y2 = 2020
year2 = pandas.read_csv(os.path.join(
    root, '{}_{}_centerline.csv'.format(river, y2)
))
out_root = os.path.join(river_root, 'bar_migration')
out_name = '{}_{}_{}_bar_migration.csv'.format(river, y1, y2)

years = y2 - y1
x = year1['easting'].values 
y = year1['northing'].values 

xn = year2['easting'].values 
yn = year2['northing'].values 

x1 = x
y1 = y

x2 = xn
y2 = yn

x1_smooth = medfilt(savgol_filter(x1, 11, 3), kernel_size=5)
y1_smooth = medfilt(savgol_filter(y1, 11, 3), kernel_size=5)

x2_smooth = medfilt(savgol_filter(x2, 11, 3), kernel_size=5)
y2_smooth = medfilt(savgol_filter(y2, 11, 3), kernel_size=5)

p, q, sm = correlate_clines(
    x1_smooth, 
    x2_smooth,
    y1_smooth,
    y2_smooth,
    0
)

for two, one in zip(p, q):
    plt.plot(
        [x1_smooth[one], x2_smooth[two]], 
        [y1_smooth[one], y2_smooth[two]]
    )
plt.plot(x, y, color='red', lw=2)
plt.plot(xn, yn, color='green', lw=2)
plt.savefig('MigrMatching.pdf')
plt.show()

i = pick_cutoffs(x, y, xn, yn)

# plt.scatter(x, y)
# plt.scatter(xn, yn)
# plt.show()

# Get point MR and curvature
migr_rate, migr_sign, p, q = ca.get_migr_rate(x,y,xn,yn,years,0)
migr_rate = medfilt(savgol_filter(migr_rate,11,3),kernel_size=5) # smoothing
curv,s = ca.compute_curvature(x,y)
curv = medfilt(savgol_filter(curv,71,3),kernel_size=5) # smoothing

year1['Mr_myr'] = migr_rate
year1['curv'] = curv 
year1['cutoff'] = 0
for c in i:
    year1.at[c[0]:c[1], 'Mr_myr'] = None
    year1.at[c[0]:c[1], 'cutoff'] = 1

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
}
for i0 , i1 in zip(sign, sign[1:]):
    bar = year1.iloc[i0:i1]
    if len(bar) <= 5:
        continue
    data['easting'].append(bar.mean()['easting'])
    data['northing'].append(bar.mean()['northing'])
    data['B_median_m'].append(bar['width_m'].median())
    data['B_mean_m'].append(bar['width_m'].mean())
    data['B_std_m'].append(bar['width_m'].std())
    data['B_sem_m'].append(bar['width_m'].sem())
    data['Mr_median_my'].append(bar['Mr_myr'].abs().median())
    data['Mr_std_my'].append(bar['Mr_myr'].abs().std())
    data['Mr_sem_my'].append(bar['Mr_myr'].abs().sem())
    data['curv_mean'].append(bar['curv'].abs().mean())
    data['curv_std'].append(bar['curv'].abs().std())
    data['curv_sem'].append(bar['curv'].abs().sem())
bar_df = pandas.DataFrame(data=data)
bar_df = bar_df.dropna(
    axis=0, 
    subset=['Mr_median_my', 'B_median_m']
)

out = 'migration_11_1991_2013.csv'
year1.to_csv(out)

out = 'bar/bar_migration_11_1991_2013.csv'
bar_df.to_csv(out)
