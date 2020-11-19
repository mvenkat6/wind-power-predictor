"""
Data acquisition program
"""

__author__ = "Maitreya Venkataswamy"

import requests
import pandas as pd
from datetime import timedelta


TURBINE_DATA_URL = "https://eersc.usgs.gov/api/uswtdb/v1/turbines" + \
                   "?&t_state=in.(WA,OR)&select=t_cap,t_hh,xlong,ylat"

POWER_DATA_URL = "https://transmission.bpa.gov/Business/Operations/Wind/twndbspt.txt"


def download_turbine_data():
    """Downloads the turbine metadata from the US Wind Turbine Database"""
    # Make a request to the US Wind Turbine Database
    r = requests.get(TURBINE_DATA_URL)

    # Convert the returned JSON data to a Pandas DataFrame
    turbine_data = pd.DataFrame(r.json()).dropna()

    # Convert the turbine longitude data to be in [0, 360]
    turbine_data["xlong"] = 360. + turbine_data["xlong"]

    # Return the turbine metadata
    return turbine_data


def download_power_data():
    """Downloads the wind power data from the BPA balancing authority"""
    # Download the data from the BPA balancing authority API
    power_data = pd.read_csv(POWER_DATA_URL, skiprows=10, delimiter="\t", usecols=[0, 2])

    # Change the index of the DataFrame to the datetime
    power_data.set_index(power_data.columns[0], inplace=True)

    # Adjust the timezone from PT to UTC
    power_data.index = pd.to_datetime(power_data.index) + timedelta(hours=8)

    # Return the power data
    return power_data


class DataAcquirer:
    """A class for a data acquisition program"""

    def __init__(self, turbine_data_file=None):
        "Constructor for the DataAcquirer"
        # Dowload or load the turbine metadata
        if turbine_data_file is None:
            self.turbine_data = download_turbine_data()
        else:
            self.turbine_data = pd.read_csv(turbine_data_file, index_col=0)


def main():
    """Main program execution"""
    data_acquirer = DataAcquirer(turbine_data_file="turbine_metadata.csv")


if __name__ == "__main__":
    main()
