"""
Connections to a MongoDB database
"""

__author__ = "Maitreya Venkataswamy"

import pymongo
import bson
import pickle
import logging

import utils


# Start logger
logger = logging.Logger(__name__)
utils.setup_logger(logger, "db.log")


def _pack(arr):
    """Uses BSON to package a numpy array for insertion into a MongoDB database"""
    return bson.binary.Binary((pickle.dumps(arr, protocol=2)))


class MongoDBInterface:
    """A class for interacting with the MongoDB database"""
    def __init__(self):
        """Constructor for MongoDBInterface"""
        # Initialize the client
        logger.info("starting Python MongoDB client...")
        client = pymongo.MongoClient()
        logger.info("Python MongoDB client started")

        # Access the database
        self.db = client["wind_power_predictor"]

    def upsert_turbine_weather_data(self, date_data, data_interp, turbine_data):
        """Upserts the weather data at the turbine locations for a single time"""
        # Construct the record to upsert
        record = {"time": str(date_data),
                  "u": _pack(data_interp["u"]),
                  "v": _pack(data_interp["v"]),
                  "T": _pack(data_interp["T"]),
                  "power": data_interp["power"],
                  "lat": _pack(turbine_data["ylat"].to_numpy()),
                  "lon": _pack(turbine_data["xlong"].to_numpy())}

        # Upsert the record
        res = self.db["turbine_weather"].replace_one(
            filter={"time": str(date_data)},
            replacement=record,
            upsert=True
        )
        if res.matched_count == 0:
            logger.info("inserted new turbine weather record for {}".format(date_data))
        else:
            logger.info("updated turbine weather record for {}".format(date_data))

    def upsert_weather_data(self, date_data, data):
        """Upserts the total weather data for a single time"""
        # Construct the record to upsert
        record = {"time": str(date_data),
                  "u": _pack(data["u"]),
                  "v": _pack(data["v"]),
                  "T": _pack(data["T"]),
                  "lat": _pack(data["lat"]),
                  "lon": _pack(data["lon"])}

        # Upsert the record
        res = self.db["weather"].replace_one(
            filter={"time": str(date_data)},
            replacement=record,
            upsert=True
        )
        if res.matched_count == 0:
            logger.info("inserted new weather record for {}".format(date_data))
        else:
            logger.info("updated weather record for {}".format(date_data))
