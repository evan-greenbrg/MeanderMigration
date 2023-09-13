"""
Description: A utility script to load a .pkl object generate from the make_centerline_meandering.py script.
             The pickle object is a simple class so you unfortunately have to load the entire routine before you can load the object.
Usage: This can be used to open the .pkl objects found in the Dryad data repository
"""
import pandas
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import seaborn as sns
import datetime
import os
from scipy.signal import medfilt
import functools
from scipy.optimize import bisect
from scipy import stats

sns.set_style("whitegrid")
sns.set_style("ticks")


class Centerline:
    def __init__(self, mask, crs, transform):
        self.mask = mask
        self.crs = crs
        self.transform = transform
        

    def find_image_endpoints(self, endpoints, es):
        es = [v for v in es] 
        riv_end = np.empty([2,2])
        for idx, v in enumerate(es):
            if v == 'N':
                i = np.where(endpoints[:,1] == endpoints[:,1].min())[0][0]
            elif v == 'E':
                i = np.where(endpoints[:,0] == endpoints[:,0].max())[0][0]
            elif v == 'S':
                i = np.where(endpoints[:,1] == endpoints[:,1].max())[0][0]
            elif v == 'W':
                i = np.where(endpoints[:,0] == endpoints[:,0].min())[0][0]
            riv_end[idx, :] = endpoints[i,:]

        return riv_end

    def find_intersections(self):
        rr, cc = np.where(self.mask)

        rows = []
        cols = []
        # Get neighboring pixels
        for r, c in zip(rr, cc):
            window = self.mask[r-1:r+2, c-1:c+2]

            # if len(window[window]) > 3:
            if np.sum(window) > 3:
                rows.append(r)
                cols.append(c)

        return np.array([cols, rows]).transpose()


    def find_all_endpoints(self):
        rr, cc = np.where(self.mask)

        rows = []
        cols = []
        # Get neighboring pixels
        for r, c in zip(rr, cc):
            window = self.mask[r-1:r+2, c-1:c+2]

            # if len(window[window]) < 3:
            if np.sum(window) < 3:
                rows.append(r)
                cols.append(c)

        return np.array([cols, rows]).transpose()


    def remove_small_segments(self, intersections, endpoints, thresh):
        tree = spatial.KDTree(intersections)
        costs = np.where(self.mask, 1, 1)
        removed = 0
        for point in endpoints:
            distance, i = tree.query(point)
            path, dist = graph.route_through_array(
                costs, 
                start=(point[1], point[0]),
                end=(intersections[i][1], intersections[i][0]),
                fully_connected=True
            )

            path = np.array(path)
            if dist < thresh:
                self.mask[path[:,0], path[:,1]] = False
                removed += 1
            else:
                continue
            
        self.mask[intersections[:,1], intersections[:,0]] = True

        return removed

    def filter_centerline(self, thresh=5):

        labels = measure.label(centerline.mask)
        bins = np.bincount(labels.flat)[1:] 
        filt = np.argwhere(bins >= thresh) + 1
        for f in filt:
            labels[np.where(labels == f)] = 9999
        labels[labels != 9999] = 0
        labels[labels == 9999] = 1

        self.mask = labels

    def prune_centerline(self, es, thresh=10):
        removed = 999
        endpoints = self.find_all_endpoints()
        # Find the terminal endpoints
        river_endpoints = self.find_image_endpoints(endpoints, es)
        while removed > 2:
            # Find the all endpoints in the centerline
            endpoints = self.find_all_endpoints()
            # Add an intersection
            for end in river_endpoints:
                self.mask[
                    int(end[1]-1):int(end[1]+2),
                    int(end[0]
                )] = 1
                self.mask[
                    int(end[1]), 
                    int(end[0]-1):int(end[0]+2)
                ] = 1

            # Find all intersections
            intersections = self.find_intersections()

            # Remove all the small bits
            removed = self.remove_small_segments(
                intersections, 
                endpoints,
                thresh
            )
            print(removed)

        # Remove the fake intersection created at the river ends
        for end in river_endpoints:
            self.mask[
                int(end[1]-1):int(end[1]+2),
                int(end[0]
            )] = 0
            self.mask[
                int(end[1]), 
                int(end[0]-1):int(end[0]+2)
            ] = 0

    def get_idx(self):
        self.idx = np.array(np.where(self.mask)).T

    def get_xy(self):
        self.xy = np.array(rasterio.transform.xy(
            self.transform,
            self.idx[:, 0], 
            self.idx[:, 1], 
        )).T

    def get_graph(self):

        start = 0
        end = len(self.idx)-1
        tmp = [tuple(i) for i in self.idx]

        G = nx.Graph()
        H = nx.Graph()
        for idx, row in enumerate(tmp):
            G.add_node(idx, pos=row)
            H.add_node(idx, pos=row)

        # Add all edges
        for idx, nodeA in enumerate(tmp):
            for jdx, nodeB in enumerate(tmp):
                if idx == jdx:
                    continue
                else:
                    length = np.linalg.norm(np.array(nodeA) - np.array(nodeB))
                    G.add_edge(idx, jdx, length=length)

        # Reduce number of edges so each node only has two edges
        for node in G.nodes():
            # Get all edge lengths 
            edge_lengths = np.empty((len(G.edges(node)),))
            edges = np.array(list(G.edges(node)))
            for idx, edge in enumerate(edges):
                edge_lengths[idx] = G.get_edge_data(*edge)['length']

            # Only select the two smallest lengths
            if (node == start) or (node == end):
                ks = np.argpartition(edge_lengths, 2)[:1]
            else:
                ks = np.argsort(edge_lengths)[:2]

            use_edges = [tuple(i) for i in edges[ks]]

            # Add the filtered edges to the H network
            for edge in use_edges:
                length = G.get_edge_data(*edge)['length']
                H.add_edge(*edge, length=length)

        self.graph = JoinComponents(H, G)

    def graph_sort(self, es):

        if es == 'EW':
            start = np.where(self.xy[:, 0] == self.xy[:, 0].min())[0][0]
            end = np.where(self.xy[:, 0] == self.xy[:, 0].max())[0][0]

        if es == 'NS':
            start = np.where(self.xy[:, 1] == self.xy[:, 1].min())[0][0]
            end = np.where(self.xy[:, 1] == self.xy[:, 1].max())[0][0]

        # Sort the shuffled DataFrame
        path = np.array(
            nx.shortest_path(self.graph, source=start, target=end, weight='length')
        )

        self.xy = self.xy[path]
        self.graph = self.graph.subgraph(path)
        mapping = dict(zip(path, range(0, len(path))))
        self.graph = nx.relabel_nodes(self.graph, mapping)

    def smooth_coordinates(self, window=5, poly=1):
        smoothed = savgol_filter(
            (self.xy[:, 0], self.xy[:, 1]), 
            window, 
            poly
        ).transpose()

        return np.vstack([smoothed[:, 0], smoothed[:, 1]]).T

    def manually_clean(self):
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(self.mask)
        t = plt.gca()
        PD = pickData(t, self.mask)
    
        axclear = plt.axes([0.0, 0.0, 0.1, 0.1])
        bclear = Button(plt.gca(), 'Clear')
        bclear.on_clicked(PD.clear)

        axremove = plt.axes([0.1, 0.0, 0.1, 0.1])
        bremove = Button(plt.gca(), 'Remove')
        bremove.on_clicked(PD.remove)

        axdone = plt.axes([0.2, 0.0, 0.1, 0.1])
        bdone = Button(plt.gca(), 'Done')
        bdone.on_clicked(PD.done)

        fig.canvas.mpl_connect('button_press_event', PD)

        im.set_picker(5) # Tolerance in points

        plt.show()

        self.centerline_clean = PD.centerline_mask


root = 'Apalachicola_River/centerline'
n1987 = os.path.join(root, 'Apalachicola_River_1987_centerline.pkl')
n2021 = os.path.join(root, 'Apalachicola_River_2021_centerline.pkl')

with open(n1987, 'rb') as f:
    cl_year1 = pickle.load(f)

with open(n2021, 'rb') as f:
    cl_year2 = pickle.load(f)

pandas.DataFrame(cl_year1.xy, columns=['easting', 'northing']).to_csv(
    os.path.join(root, 'Apalachicola_river_1987_centerline_idx.csv')
)
pandas.DataFrame(cl_year2.xy, columns=['easting', 'northing']).to_csv(
    os.path.join(root, 'Apalachicola_river_2021_centerline_idx.csv')
)
