# Forte Sept. 2023

from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os



def transectTimeSeries(transectCsvFile,  outputFolder, rollingPeriod=6):
    """designed for coastsat output transect csv file containing
    shoreline position and date. function opens csv file and makes
    a series of plots of shoreline relative position and date.
    rolling period can be adjusted but defaults to 6 records"""

    # read transectCsvFile
    df = pd.read_csv(transectCsvFile, header=0, index_col=False)

    # drop last zeros off date for conversion - make dates for plot
    df['dates'] = df['dates'].str[:-6]
    df['dates'] = pd.to_datetime(df['dates']).dt.strftime('%Y-%m-%d %H:%M:%S')
    dateNumbers = matplotlib.dates.date2num(df['dates'])

    # set plot font sizes
    plt.rcParams.update({'font.size': 18})

    #id columns for iteration
    cols = df.columns
    for d in cols[2::]:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(dateNumbers, df[d], '.', alpha=0.2)
        ax.plot(dateNumbers, df[d].rolling(rollingPeriod, min_periods=2).mean())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel('Year')
        ax.set_ylabel('Relative Shoreline Position (m)')
        ax.set_title(d)
        plt.grid()
        plt.savefig(outputFolder + d + '_timeSeriesPlot.png')
        plt.close()




# Perhasp Can reduce regressionShapefileName and output folder from parameters
def transectRegression(transectCsvFile, shorelineFile, transectsFile, regressionShapefileName, outputFolder):
    """inputs: transectCsvFile - transect file output from coastsat
                shorelineFile - takes the final .shp file of coastsat generated shorelines
                transectsFile - the transect file generated to extract relative shoreline distance values with each shoreline
                regressionShapefileName: output shape file name
                outputFolder: where to store output files
        function reads in the files and calculates a linear regression (LR) at each transect location
        finds middle shoreline (from all coastsat shorelines) to plot the LR coefficient spatially
        makes spatial plot with aerial imagery as background with LR at each locations
        ** needs some error handling**
    """
    df = pd.read_csv(transectCsvFile, header=0, index_col=False)
    df = df.dropna(axis=0, how='any')
    df['dates'] = pd.to_datetime(df['dates'])
    dateRange = df['dates'].max() - df['dates'].min()
    timeInDays = dateRange.days
    cols = df.columns
    rates = []
    rsquare = []

    timeDistance = pd.Series(np.arange(1, timeInDays, timeInDays / len(df)))
    timeDistance = timeDistance.values.reshape(-1, 1)

    plt.rcParams.update({'font.size': 18})
    for t in cols[2::]:
        ychange = df[t]
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(timeDistance, ychange)  # perform linear regression
        Y_pred = linear_regressor.predict(timeDistance)
        rate = linear_regressor.coef_  # linear regression coefficient
        rates.append(rate)
        score = linear_regressor.score(timeDistance, ychange)  # rsquared value
        rsquare.append(score)
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(timeDistance, ychange)
        ax.plot(timeDistance, Y_pred, color='red')
        ax.set_title('{}              rate: {} '.format(t, rate))
        ax.set_xlabel('Days')
        ax.set_ylabel('Relative Shoreline Position (m)')
        plt.grid()
        plt.savefig(outputFolder + t + '_Regression.png')
        plt.close()
    ratesArray = np.array(rates)
    rsquareArray = np.array(rsquare)

    s = gpd.read_file(shorelineFile)
    s['shorelineLengths'] = s.geometry.length
    t = gpd.read_file(transectsFile)
    shorelineNumber = int(len(s) / 2)
    median = np.median(s['shorelineLengths'])
    middle = s.iloc[[shorelineNumber]]  # find middle shoreline
    if middle['shorelineLengths'].values >= median:
        middle = s.iloc[[shorelineNumber]]
    else:
        middle = s.iloc[[shorelineNumber + 3]]

    points = t.unary_union.intersection(middle.unary_union)  # find where transects cross middle shoreline

    crossPoints = gpd.GeoSeries(points)
    cp = crossPoints.explode(index_parts=True)  #setting up new geodataframe
    zz = gpd.GeoDataFrame(gpd.GeoSeries(cp))
    zz = zz.reset_index(drop=True)
    zz = zz.set_geometry(zz[0])
    zz = zz.drop(columns=[0])
    # need to ensure lengths are correct of regression values and transect numbers
    zz['regress'] = ratesArray * 365.2425  # add linear regression values and convert daily rates to annual rates
    zz['rSquared'] = rsquareArray  # add rsquared values
    zz.to_file(regressionShapefileName, crs=s.crs.to_epsg())  # need file to write out final shapefile
    zz = zz.set_crs(s.crs.to_epsg())

    margin = 300  # expands plotting bounds (meters)
    xMin, yMin, xMax, yMax = list(zz['geometry'].total_bounds)
    xMin = xMin - margin
    xMax = xMax + margin
    yMin = yMin - margin
    yMax = yMax + margin


    fig, ax = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    zz.plot(ax=ax, markersize=np.abs(zz['regress']) * 50, column='regress', cmap='seismic_r', legend=True, cax=cax,
            legend_kwds={'label': 'm/year', "orientation": "horizontal", 'shrink': 0.5})
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ctx.add_basemap(ax, crs=zz.crs, source=ctx.providers.Esri.WorldImagery, attribution_size=2, reset_extent=False)
    ax.set_title('LRR [2008 - 2023] Satellite Derived Shorelines')
    ax.set_axis_off()
    fig.savefig(outputFolder + 'LRR.png')
















