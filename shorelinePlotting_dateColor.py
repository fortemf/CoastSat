import os
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import datetime
from matplotlib.dates import DateFormatter
import rasterio
from rasterio import plot as rasterplot
from matplotlib.colors import TwoSlopeNorm
import geopandas as gpd



file = r'C:\CoastSat\FY23\shorelines\toMike\TESTARC5_output-.shp'
tiff = r'C:\CoastSat\FY23\shorelines\toMike\shorelineTesting.tif'

raster = rasterio.open(tiff)
tiff_extent = [raster.bounds[0], raster.bounds[2], raster.bounds[1], raster.bounds[3]]

data = gpd.read_file(file)

dateObjects = []
for val in data['date']:
    dateObjects.append(datetime.datetime.strptime(val, '%Y-%m-%d'))

data['datenumber'] = matplotlib.dates.date2num(dateObjects)


vmin, vmax = data['datenumber'].min(), data['datenumber'].max()
vcenter = np.median(data['datenumber'])
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

cmap = 'cool'
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)



f, ax = plt.subplots()
rasterplot.show(raster,extent=tiff_extent, ax=ax)
data.plot(column='datenumber',cmap=cmap,norm=norm,legend=False, ax=ax)
cbr = f.colorbar(cbar,ax=ax, format=DateFormatter('%Y %m-%d'))
cbr.ax.tick_params(labelsize=16)



ax.set_title('Satellite derived shorelines', fontsize=16)
ax.set_ylabel('UTM N (m)', fontsize=16)
ax.set_xlabel('UTM E (m)', fontsize=16)




#arcpro information - https://pro.arcgis.com/en/pro-app/latest/tool-reference/data-management/apply-symbology-from-layer.htm

# below line changes layer symbology using keyword UPDATE to update the fields/
# arcpy.management.ApplySymbologyFromLayer('TESTARC5_output-.shp',in_symbology_layer,'','UPDATE')