"""
Data acquisition program
"""

__author__ = "Maitreya Venkataswamy"

import requests
import numpy as np
import pandas as pd
import logging
import time
from datetime import timedelta, datetime
from dateutil import rrule
from functools import lru_cache
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


from database import MongoDBInterface
import utils

TURBINE_DATA_URL = "https://eersc.usgs.gov/api/uswtdb/v1/turbines" + \
                   "?&t_state=in.(WA,OR)&select=t_cap,t_hh,xlong,ylat"

POWER_DATA_URL = "https://transmission.bpa.gov/Business/Operations/Wind/twndbspt.txt"

WEATHER_DATA_URL = "https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs{}/gfs_0p25_1hr_{:02d}z"

HEIGHT_VEL = np.array([10, 20, 30, 40, 50, 80, 100])
HEIGHT_T = np.array([2, 80, 100])

LON_MIN = 360. - 135.
LON_MAX = 360. - 105.
LAT_MIN = 35.
LAT_MAX = 55.


# Start logger
logger = logging.Logger(__name__)
utils.setup_logger(logger, "data.log")


def _download_turbine_data():
    """Downloads the turbine metadata from the US Wind Turbine Database"""
    # Make a request to the US Wind Turbine Database
    r = requests.get(TURBINE_DATA_URL)

    # Convert the returned JSON data to a Pandas DataFrame
    turbine_data = pd.DataFrame(r.json()).dropna()

    # Convert the turbine longitude data to be in [0, 360]
    turbine_data["xlong"] = 360. + turbine_data["xlong"]

    # Save the turbine data to disk
    turbine_data.to_csv("turbine_metadata.csv")

    # Return the turbine metadata
    return turbine_data


@lru_cache(maxsize=1)
def _download_power_data(date, hr):
    """Downloads the wind power data from the BPA balancing authority"""
    # Download the data from the BPA balancing authority API
    power_data = pd.read_csv(POWER_DATA_URL, skiprows=10, delimiter="\t", usecols=[0, 2])

    # Change the index of the DataFrame to the datetime
    power_data.set_index(power_data.columns[0], inplace=True)

    # Adjust the timezone from PT to UTC
    power_data.index = pd.to_datetime(power_data.index) + timedelta(hours=8)

    # Return the power data
    return power_data


def _get_power_data(date_time):
    """Finds the wind power generation for a date and hour"""
    # Download the power data using the helper function
    power_data = _download_power_data(datetime.now().date(), datetime.now().hour)

    # Find and return the power data
    if date_time in power_data.index:
        return power_data.loc[date_time]["Wind"]
    else:
        return np.nan


@lru_cache(maxsize=16)
def _download_weather_data(date, hr_base):
    """Downloads a single weather forecast dataset from the GFS system"""
    # Get the date as a string formatted as YYYYmmdd
    date_str = date.strftime("%Y%m%d")

    # Download and return the dataset
    try:
        return Dataset(WEATHER_DATA_URL.format(date_str, hr_base))
    except OSError:
        logger.warning("data for {}, hr_base={} is not available yet".format(date_str, hr_base))
        return None


@lru_cache(maxsize=128)
def _get_weather_data(date, hr, hr_base):
    """Get the weather data for a specific day and hour"""
    # Download the raw data frorm the GFS system in NetCDF4 format
    weather_data = _download_weather_data(date, hr_base)

    # Check if the data was downloaded correctly
    if weather_data is None:
        _download_weather_data.cache_clear()
        return None

    # Get the variables from the weather data
    vars = weather_data.variables

    # Initialize a dictionary to hold the data
    data = {}

    # Attempt to access the data
    try:
        # Get the latitude and longitude data
        lon = vars["lon"][:].filled()
        lat = vars["lat"][:].filled()

        # Get the indices of the Pacific-Northwest region of the data
        lat_idx = np.logical_and(LAT_MIN <= lat, lat <= LAT_MAX)
        lon_idx = np.logical_and(LON_MIN <= lon, lon <= LON_MAX)
        idx = np.ix_(lat_idx, lon_idx)

        # Save the latitude and longitude data
        data["lat"] = lat[lat_idx]
        data["lon"] = lon[lon_idx]

        # Extract the velocity data
        data["u"] = np.zeros((np.sum(lat_idx), np.sum(lon_idx), len(HEIGHT_VEL)))
        data["v"] = np.zeros((np.sum(lat_idx), np.sum(lon_idx), len(HEIGHT_VEL)))
        for k, h in enumerate(HEIGHT_VEL):
            data["u"][:, :, k] = vars["ugrd{}m".format(h)][hr, :, :].filled()[idx]
            data["v"][:, :, k] = vars["vgrd{}m".format(h)][hr, :, :].filled()[idx]

        # Extract the temperature data
        data["T"] = np.zeros((np.sum(lat_idx), np.sum(lon_idx), len(HEIGHT_T)))
        for k, h in enumerate(HEIGHT_T):
            data["T"][:, :, k] = vars["tmp{}m".format(h)][hr, :, :].filled()[idx]

        # Compute the streamlines
        skp = 8
        sp = plt.streamplot(data["lon"][::skp], data["lat"][::skp],
                            data["u"][::skp, ::skp, 0], data["v"][::skp, ::skp, 0],
                            density=4)
        lines = sp.lines

        # Combine the streamlines into a single list for plotting
        pts = []
        for pt in lines.get_segments():
            pts.append(pt)
            pts.append(np.array([np.nan, np.nan]))
        pts = np.vstack(pts)

        # Add the streamline points to the data
        data["pts"] = pts

        # Return the data
        return data
    except Exception:
        logger.error("cannot access variables from the downloaded data")
        return None


def _interp_data(turbine_data, var, lat, lon, height):
    """Interpolates a weather data variable to the turbine location"""
    # Define a trilinear interpolator on a regular grid
    rgi = RegularGridInterpolator((lat, lon, height,), var, method="linear")

    # Assemble the query points using the turbine hub locations
    turb_locs = turbine_data[["ylat", "xlong", "t_hh"]].to_numpy()

    # Return the interpolated data
    return rgi(turb_locs)


class DataAcquirer:
    """A class for a data acquisition program"""

    def __init__(self, turbine_data_file=None, initial_days=5):
        "Constructor for the DataAcquirer"
        # Dowload or load the turbine metadata
        if turbine_data_file is None:
            self.turbine_data = _download_turbine_data()
        else:
            self.turbine_data = pd.read_csv(turbine_data_file, index_col=0)

        # Initialize the MongoDB interface
        self.mongodb_interface = MongoDBInterface()

        # Download and process the initial data
        start_date = datetime.utcnow().date() - timedelta(days=initial_days)
        end_date = datetime.utcnow().date()
        self._acquire_date_range(start_date, end_date)

        # Begin the incremental download loop
        self._incremental_acquisition_loop()

    def _download_and_process_data(self, date_time, hr, hr_base):
        """Downloads and processes a single dataset"""
        # Determine the date and time of the data
        date_data = date_time + timedelta(hours=hr_base + hr)

        # TEMP:
        logger.info("downloading data for {} using data from {}, base hour {}, hour {}".format(
                    date_data, date_time.date(), hr_base, hr))

        # Get the weather data
        data = _get_weather_data(date_time.date(), hr, hr_base)

        # Check if the data was successfully downloaded
        if data is None:
            _get_weather_data.cache_clear()
            return False

        # Upsert the weather data
        if not self.mongodb_interface.upsert_weather_data(date_data, data):
            return False

        # Allocate a dictionary for the interpolated data
        data_interp = {}

        # Interpolate the weather data to the turbine locations
        for var, height in zip(["u", "v", "T"], [HEIGHT_VEL, HEIGHT_VEL, HEIGHT_T]):
            data_interp[var] = _interp_data(self.turbine_data, data[var],
                                            data["lat"], data["lon"], height)

        # Get the power data
        data_interp["power"] = _get_power_data(date_data)

        # Upsert the interpolated data
        if not self.mongodb_interface.upsert_turbine_weather_data(date_data, data_interp,
                                                                  self.turbine_data):
            return False

        # Return True to indicate success
        return True

    def _acquire_date_range(self, start_date, end_date, hrs_from_sim=24):
        """Downloads and processes data in a date range"""
        # Iterate over the days in the range
        for date_time in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
            # Iterate over the four simulations in the day
            for hr_base in [0, 6, 12, 18]:
                # Iterate over the hours to use from each simulation
                for hr in range(hrs_from_sim):
                    # Download and process the data
                    if not self._download_and_process_data(date_time, hr, hr_base):
                        break

    def _incremental_acquisition_loop(self, hrs_wait=1):
        """Infinte loop that incrementally downloads and processes new data"""
        while True:
            # Attempt to download all the data for today
            logger.info("performing incremental download...")
            self._acquire_date_range(datetime.utcnow().date(),
                                     datetime.utcnow().date() + timedelta(days=1),
                                     hrs_from_sim=24)

            # Wait and try again
            logger.info("waiting {} hour(s) until next incremental download...".format(hrs_wait))
            time.sleep(hrs_wait * 3600)


def main():
    """Main program execution"""
    DataAcquirer(turbine_data_file="turbine_metadata.csv")


if __name__ == "__main__":
    main()
