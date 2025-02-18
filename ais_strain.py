#!/usr/local/Caskroom/miniconda/base/envs/fresh/bin/python

'''
By Chance

Loads velocity data and calculates strain rates

Input: .ini config file which fills all variables in preamble

Output: a geoTIFF file


'''

# System libraries
import os
import sys
import h5py
import configparser
import json

#Standard libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio as rs
import rioxarray as rx

#For geometries
import shapely
from shapely import box, LineString, MultiLineString, Point, Polygon, LinearRing
from shapely.geometry.polygon import orient

#For REMA
from rasterio import plot
from rasterio.mask import mask
from rasterio.features import rasterize

#Datetime
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from dateutil.relativedelta import relativedelta
import time

#For plotting, ticking, and line collection
from matplotlib import cm 
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
import cmcrameri.cm as cmc
import contextily as cx
import earthpy.spatial as es
# for legend
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple

#For error handling
import shutil
import traceback

#For raster
from rasterio.transform import from_origin

# not in use 
from ipyleaflet import Map, basemaps, Polyline, GeoData, LayersControl
from rasterio import warp
from rasterio.crs import CRS


####################################################################################################################
##CONFIG##
####################################################################################################################

#config stuff

gdal_data = '/usr/local/Caskroom/miniconda/base/envs/fresh/share/gdal'
proj_lib = '/usr/local/Caskroom/miniconda/base/envs/fresh/share/proj'
proj_data = '/usr/local/Caskroom/miniconda/base/envs/fresh/share/proj'

# choose file

where = '/Volumes/nox2/Chance/its_live/'
file_ux = f'{where}/ITS_LIVE_velocity_120m_RGI19A_2019_v02_vx.tif'
file_uy = f'{where}/ITS_LIVE_velocity_120m_RGI19A_2019_v02_vy.tif'

# Crop shape (not currently in use)
shape = 'shapes/scripps_antarctic_polygons_CR.shp'


# output file

out_where = '/Volumes/nox2/Chance/its_live/'
out_file = f'{out_where}/its_live_2019_e1.tif'


####################################################################################################################
##CODE##
####################################################################################################################

crs_antarctica = 'EPSG:3031'
crs_latlon = 'EPSG:4326'

dsux = rx.open_rasterio(file_ux)  # Open with rioxarray 
tr = dsux.rio.transform()

dsux = xr.open_dataset(file_ux)
dsux.rio.set_spatial_dims(x_dim='y', y_dim='x', inplace=True)
dsux.rio.write_crs(crs_antarctica, inplace=True)

dsuy = xr.open_dataset(file_uy)
dsuy.rio.set_spatial_dims(x_dim='y', y_dim='x', inplace=True)
dsuy.rio.write_crs(crs_antarctica, inplace=True)

#u=xr.merge([dsux.rename_vars({'band_data': 'ux'}), dsuy.rename_vars({'band_data': 'uy'})]).sel(band=1).drop_vars(['spatial_ref', 'band'])
u=xr.merge([dsux.rename_vars({'vx': 'ux'}), dsuy.rename_vars({'vy': 'uy'})]).sel(band=1).drop_vars(['band'])
u.rio.set_spatial_dims(x_dim='y', y_dim='x', inplace=True)
u.rio.write_crs(crs_antarctica, inplace=True);


#shapes

#roi = 'shapes/I-Ipp_larsenc_mélange.shp'

try:
    gdf = gpd.read_file(shape).set_crs(crs_latlon, allow_override=True).to_crs(crs_antarctica)
except: 
    print('no shape file found, moving on')
#gdf = gdf[gdf.Regions=='East']

#roi_gdf = gpd.read_file(roi).set_crs(crs_latlon, allow_override=True).to_crs(crs_antarctica)


# run

x, y = u.x.values, u.y.values

u_0_0 = u.ux.differentiate(coord='x').values
u_0_1 = u.ux.differentiate(coord='y').values
u_1_0 = u.uy.differentiate(coord='x').values
u_1_1 = u.uy.differentiate(coord='y').values

# numpy arrays are denoted with capitals
# xarrays in lowercase with indices in the name

# Jacobian
U = np.array([[u_0_0, u_0_1], [u_1_0, u_1_1]])
# Pauli z matrix
pauli_3 = np.array([[1, 0], [0 , -1]])
# strain tensor
E = 0.5*(U + U.transpose(1, 0, 2, 3))
# 2D Rotation matrix and a function
R = lambda theta: np.array([[np.cos(theta*np.pi/180), -np.sin(theta*np.pi/180)], [np.sin(theta*np.pi/180), np.cos(theta*np.pi/180)]])
def rotate(T, theta): return np.einsum('ki,klmn,lj', R(theta), T, R(theta))
# Rotated strain tensor E'
E_prime = lambda theta: rotate(E, theta)

# strain tensor in x, y reference frame
e_ij = xr.DataArray(E_prime(0).transpose(3, 2, 0, 1), coords={'x': x, 'y': y}, dims=['x', 'y', 'i', 'j'])
# Principal strain rates
# mohr circle radius
r = np.sqrt((0.5*np.einsum('ij,ijmn', pauli_3, E))**2 + E[0, 1, :, :]**2)
# mohr circle center
c = 0.5*np.einsum('iimn', E)
# principle strain rates
E_p = np.einsum('imn,ij->ijmn', c - np.einsum('mn,ii->imn', r, pauli_3), np.identity(2))
'''
E_d = E_p - (1/2)*np.einsum('iimn', E_p)
e_ij_p = xr.DataArray(E_p.transpose(2, 3, 0, 1), coords={'y': y, 'x': x}, dims=['y', 'x', 'i', 'j'])
e_ij_d = xr.DataArray(E_d.transpose(2, 3, 0, 1), coords={'y': y, 'x': x}, dims=['y', 'x', 'i', 'j'])
# angles from EPSG:3031 reference frame
theta_p = xr.DataArray(0.5*np.arctan2(E[0, 1, :, :], (np.einsum('ij,ijmn', pauli_3, E))), coords={'y': y, 'x': x})
u_mag = xr.DataArray(np.sqrt(u.ux**2 + u.uy**2), coords={'y': y, 'x': x})

# Get flow direction from velocities
theta_u = np.arctan2(u.uy, u.ux)
# Rotated train tensor DataArray
E_u = np.einsum('kimn,klmn,ljmn->ijmn', R(theta_u), E, R(theta_u))
e_ij_u = xr.DataArray(E_u.transpose(2, 3, 0, 1), coords={'y': y, 'x': x}, dims=['y', 'x', 'i', 'j'])

# set up with rio for plottting
e_ij_p.rio.set_spatial_dims(x_dim='y', y_dim='x', inplace=True);
e_ij_p.rio.write_crs(crs_antarctica, inplace=True);
e_ij_d.rio.set_spatial_dims(x_dim='y', y_dim='x', inplace=True);
e_ij_d.rio.write_crs(crs_antarctica, inplace=True);
e_ij_u.rio.set_spatial_dims(x_dim='y', y_dim='x', inplace=True);
e_ij_u.rio.write_crs(crs_antarctica, inplace=True);
u_mag.rio.set_spatial_dims(x_dim='y', y_dim='x', inplace=True);
u_mag.rio.write_crs(crs_antarctica, inplace=True);
'''


## save

height, width = E_p[0, 0, :, :].shape  # Ensure these are integers

meta = {
    "driver": "GTiff",
    "height": int(height),
    "width": int(width),
    "count": 1,  # Assuming a single-band raster
    "dtype": E_p.dtype,
    "crs": "EPSG:3031",  # Replace with actual CRS if available
    "transform": tr,
}

with rs.open(out_file, "w", **meta) as dst:
    dst.write(E_p[1, 1, :, :], 1)