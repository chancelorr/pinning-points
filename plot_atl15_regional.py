#!/usr/local/Caskroom/miniconda/base/envs/fresh/bin/python

'''
By Chance

Loads and processes the data

Input: .ini config file which fills all variables in preamble

Output: a bunch of figures


'''
# System libraries
import os
import sys
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
import matplotlib.patheffects as path_effects
import cmcrameri.cm as cmc
import contextily as cx
import earthpy.spatial as es
# for legend
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple

#Personal and application specific utilities
from utils.nsidc import download_is2
#from utils.S2 import plotS2cloudfree, add_inset, convert_time_to_string
from utils.utilities import is2dt2str
import pyTMD

#For error handling
import shutil
import traceback

#For raster
from rasterio.transform import from_origin

# not in use 
from ipyleaflet import Map, basemaps, Polyline, GeoData, LayersControl
from rasterio import warp
from rasterio.crs import CRS


### Setup from config

ini = 'config/ATL15.ini'

######## Load variables ###########
# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the configuration file
config.read(ini)

#os and pyproj paths
gdal_data = config.get('os', 'gdal_data')
proj_lib = config.get('os', 'proj_lib')
proj_data = config.get('os', 'proj_data')

#path params
shape = f'shapes/scripps_antarctic_polygons_CR.shp'
output_dir = config.get('data', 'output_dir')
rema_path = config.get('data', 'rema_path')
try: plot_dir = config.get('data', 'plot_dir')
except: plot_dir='plots'

#access params
uid = config.get('access', 'uid')
pwd = config.get('access', 'pwd')
email = config.get('access', 'email')

#Print results
os.environ["GDAL_DATA"] = gdal_data # need to specify to make gdal work
os.environ["PROJ_LIB"] = proj_lib # need to specify to make pyproj work
os.environ["PROJ_DATA"] = proj_data # need to specify to make pyproj work


##### Functions

def tester(h_plot):
    print(f"{h_plot.attrs['short_name']}")

def set_axis_color(ax, axcolor):
    ax.spines['bottom'].set_color(axcolor)
    ax.spines['top'].set_color(axcolor) 
    ax.spines['right'].set_color(axcolor)
    ax.spines['left'].set_color(axcolor)
    ax.tick_params(axis='x', colors=axcolor)
    ax.tick_params(axis='y', colors=axcolor)
    ax.yaxis.label.set_color(axcolor)
    ax.xaxis.label.set_color(axcolor)
    ax.title.set_color(axcolor)

def make_map(h_plot, dpi=600, vlims=[-10, 10], save=False, transparent=True):
    fig, ax = plt.subplots(figsize=[10,13], dpi=dpi)
    cax = ax.inset_axes([1.03, 0, 0.1, 1], transform=ax.transAxes) # Need dummy axis for colorbar so I can remove it later

    major_font_size = 12
    minor_font_size = 10
    line_w = 0.1
    cmap = cmc.vik_r

    h_plot.plot.pcolormesh(ax=ax, cmap=cmap, vmin=vlims[0], vmax=vlims[1], cbar_kwargs={'cax': cax})
    cax.remove() #remove colorbar

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_title('')

    # Shapefile stuff
    gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=line_w)

    # put a patch behind colorbar so I can read it in dark mode
    pos = ax.get_position()
    cax_pos = [pos.x0+pos.width*0.17, pos.y0+pos.height*0.88, 0.25, 5e-3]
    cax = fig.add_axes(cax_pos, zorder=5)
    fig.patches.append(Rectangle((cax_pos[0]-cax_pos[0]*.05, cax_pos[1]-cax_pos[3]*9), cax_pos[2]*1.1, cax_pos[3]*11,
        transform=fig.transFigure, color='white', zorder=1))  

    # Add scalar mappable Equivalent to the above colormap
    norm = mcolors.Normalize(vmin=vlims[0], vmax=vlims[1])
    sm = cm.ScalarMappable(cmap=cmc.vik_r, norm=norm)
    sm.set_array([])

    # Add a colorbar to the plot, with a specific location
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', fraction=0.1, pad=0.0)
    cbar.ax.tick_params(labelsize=minor_font_size) 
    cbar.set_label(f"ICESat-2 ATL15 {h_plot.attrs['long_name']} \n {h_plot.time.dt.strftime('%Y-%m').values} ({h_plot.attrs['units']})", fontsize=minor_font_size)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)
    set_axis_color(cax, 'black')
    # Customize the colorbar ticks if needed
    cbar.set_ticks([-5, 0, 5])

    plotname = f"{plot_dir}/ATL15_AE_{h_plot.attrs['short_name']}_{h_plot.time.dt.strftime('%Y%m%d').values}.png"
    if save: 
        fig.savefig(plotname, dpi=dpi, bbox_inches='tight', transparent=transparent)

    plt.close(fig)

    return fig

###################### End Functions ########################


###################### Make shapes ######################
####
crs_antarctica = 'EPSG:3031'
crs_latlon = 'EPSG:4326'
short_name = 'ATL15'
# Read shapefile into gdf for everything
gdf = gpd.read_file(shape).set_crs(crs_latlon, allow_override=True).to_crs(crs_antarctica)
# Only Wilkes Land and Princess elizabeth land
gdf = gdf[(gdf.Subregion=='Cp-D')+(gdf.Subregion=='C-Cp')+(gdf.Subregion=='D-Dp')]

# Separate by entry type
gdf_fl = gdf[gdf.Id_text=='Ice shelf']
gdf_pp = gdf[(gdf.Id_text=='Ice rise or connected island')]
gdf_ext = gpd.GeoDataFrame(geometry=[gdf_fl.apply(lambda p: Polygon(p.geometry.exterior.coords), axis=1).unary_union.union(gdf_pp.unary_union)],
    crs=crs_antarctica).explode(ignore_index=True)
gdf_gr = gdf[gdf.Id==1]
gdf_ext_all = gpd.GeoDataFrame(geometry=[gdf_ext.unary_union.union(gdf_gr.unary_union)], crs=crs_antarctica)
gdf_bbox = gpd.read_file('shapes/ATL15_AE_bbox.shp')


gdf_bbox = gdf_ext_all.bounds
buffer = (gdf_bbox.minx - gdf_bbox.miny)*0.02
gdf_bbox = gpd.GeoDataFrame(geometry=[box(gdf_bbox.minx-buffer, gdf_bbox.miny-buffer, gdf_bbox.maxx+buffer, gdf_bbox.maxy+buffer)[0]])

###################### Load Data ######################

processed_dir='/Volumes/nox2/Chance/processed_data/'
print('Reading netCDFs into XR datasets...', end='', flush=True)
ae_dhdt_lag1 = xr.open_dataset(f'{processed_dir}/processed_ATL15_AE.nc', group='/dhdt_lag1/')
ae_dhdt_lag4 = xr.open_dataset(f'{processed_dir}/processed_ATL15_AE.nc', group='/dhdt_lag4/')
ae_dhdt_lag20 = xr.open_dataset(f'{processed_dir}/processed_ATL15_AE.nc', group='/dhdt_lag20/')
ae_dh = xr.open_dataset(f'{processed_dir}/processed_ATL15_AE.nc', group='/delta_h/')

# set up in rioxarray
ae_dhdt_lag1.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
ae_dhdt_lag4.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
ae_dhdt_lag20.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
ae_dh.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)

ae_dhdt_lag1.rio.write_crs(crs_antarctica, inplace=True)
ae_dhdt_lag4.rio.write_crs(crs_antarctica, inplace=True)
ae_dhdt_lag20.rio.write_crs(crs_antarctica, inplace=True)
ae_dh.rio.write_crs(crs_antarctica, inplace=True)

attr_list = ['long_name', 'short_name', 'units', 'resolution']
ae_dhdt_lag1.dhdt.attrs[attr_list[0]], ae_dhdt_lag1.dhdt.attrs[attr_list[1]], ae_dhdt_lag1.dhdt.attrs[attr_list[2]], ae_dhdt_lag1.dhdt.attrs[attr_list[3]] = 'Quarterly dh/dt', 'dhdt_lag1', 'm yr$^{{-1}}$', '1 km'
ae_dhdt_lag4.dhdt.attrs[attr_list[0]], ae_dhdt_lag4.dhdt.attrs[attr_list[1]], ae_dhdt_lag4.dhdt.attrs[attr_list[2]], ae_dhdt_lag4.dhdt.attrs[attr_list[3]] = 'Annual dh/dt', 'dhdt_lag4', 'm yr$^{{-1}}$', '1 km'
ae_dhdt_lag20.dhdt.attrs[attr_list[0]], ae_dhdt_lag20.dhdt.attrs[attr_list[1]], ae_dhdt_lag20.dhdt.attrs[attr_list[2]], ae_dhdt_lag20.dhdt.attrs[attr_list[3]] = 'Pentennial dh/dt', 'dhdt_lag20', 'm yr$^{{-1}}$', '1 km'
ae_dh.delta_h.attrs[attr_list[0]], ae_dh.delta_h.attrs[attr_list[1]], ae_dh.delta_h.attrs[attr_list[2]], ae_dh.delta_h.attrs[attr_list[3]] = 'Height Change', 'delta_h', 'm', '1 km'

print('DONE')

###################### Editable Code ######################


###################### Make Maps ######################

# Loop through variables and make map


for ds in [ae_dhdt_lag1.dhdt, ae_dhdt_lag4.dhdt, ae_dhdt_lag20.dhdt, ae_dh.delta_h]:
    for t in ds.time.values:
        h_plot = ds.sel(time=t)
        h_plot = h_plot.rio.clip(gdf_ext_all.geometry.values, crs=crs_antarctica, drop=True)
     
        print(f"Plotting {h_plot.attrs['short_name']}, {h_plot.time.dt.strftime('%Y-%m').values}...", end="", flush=True)
        fig = make_map(h_plot, dpi=600, vlims=[-10, 10], save=True, transparent=True)
        print('DONE')