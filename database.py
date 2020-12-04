"""
Connections to a MongoDB database
"""

__author__ = "Maitreya Venkataswamy"

import pymongo
import bson
import pickle
import logging

import utils

ATLS_STR = "mongodb+srv://dash-app:wind-power@wind-power-predictor.yzev6.mongodb.net/<dbname>?retryWrites=true&w=majority"


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
        client = pymongo.MongoClient(ATLAS_STR)
        logger.info("Python MongoDB client started")

        # Access the database
        self.db = client["wind_power_predictor"]

    def upsert_turbine_weather_data(self, date_data, data_interp, turbine_data):
        """Upserts the weather data at the turbine locations for a single time"""
        try:
            # Construct the record to upsert
            record = {"time": date_data,
                      "u": _pack(data_interp["u"]),
                      "v": _pack(data_interp["v"]),
                      "T": _pack(data_interp["T"]),
                      "power": data_interp["power"],
                      "lat": _pack(turbine_data["ylat"].to_numpy()),
                      "lon": _pack(turbine_data["xlong"].to_numpy())}

            # Upsert the record
            res = self.db["turbine_weather"].replace_one(
                filter={"time": date_data},
                replacement=record,
                upsert=True
            )
            if res.matched_count == 0:
                logger.info("inserted new turbine weather record for {}".format(date_data))
            else:
                logger.info("updated turbine weather record for {}".format(date_data))

            return True
        except Exception:
            logger.error("failed to upsert new turbine weather record for {}".format(date_data))
            return False

    def upsert_weather_data(self, date_data, data):
        """Upserts the total weather data for a single time"""
        try:
            # Construct the record to upsert
            record = {"time": date_data,
                      #"u": _pack(data["u"]),
                      #"v": _pack(data["v"]),
                      #"T": _pack(data["T"]),
                      "pts": _pack(data["pts"]),
                      #"lat": _pack(data["lat"]),
                      #"lon": _pack(data["lon"])
                      }

            # Upsert the record
            res = self.db["weather"].replace_one(
                filter={"time": date_data},
                replacement=record,
                upsert=True
            )
            if res.matched_count == 0:
                logger.info("inserted new weather record for {}".format(date_data))
            else:
                logger.info("updated weather record for {}".format(date_data))

            return True
        except Exception:
            logger.error("failed to upsert new weather record for {}".format(date_data))
            return False

    def fetch_turbine_weather_data(self, start_date_time, end_date_time):
        """Fetches turbine weather data in a date range from the database"""
        # Allocate a list for the data
        data = []

        # Aseemble a query
        query = {"$and": [{"time": {"$gt": start_date_time}},
                          {"time": {"$lt": end_date_time}}]}

        # Query the database and parse the results
        for record in self.db["turbine_weather"].find(query):
            # Allocate a dictionary for the single record
            rec = {}

            # Parse the record into the data
            rec["time"] = record["time"]
            rec["u"] = pickle.loads(record["u"])
            rec["v"] = pickle.loads(record["v"])
            rec["T"] = pickle.loads(record["T"])
            rec["power"] = record["power"]
            rec["lat"] = pickle.loads(record["lat"])
            rec["lon"] = pickle.loads(record["lon"])

            # Add the parsed record to the data list
            data.append(rec)

        # Sort the records
        data.sort(key=lambda rec: rec["time"])

        # Return the data
        return data

    def fetch_weather_data(self, start_date_time, end_date_time):
        """Fetches weather data in a date range from the database"""
        # Allocate a list for the data
        data = []

        # Aseemble a query
        query = {"$and": [{"time": {"$gt": start_date_time}},
                          {"time": {"$lt": end_date_time}}]}

        # Query the database and parse the results
        for record in self.db["weather"].find(query):
            # Allocate a dictionary for the single record
            rec = {}

            # Parse the record into the data
            rec["time"] = record["time"]
            #rec["u"] = pickle.loads(record["u"])
            #rec["v"] = pickle.loads(record["v"])
            #rec["T"] = pickle.loads(record["T"])
            rec["pts"] = pickle.loads(record["pts"])
            #rec["lat"] = pickle.loads(record["lat"])
            #rec["lon"] = pickle.loads(record["lon"])

            # Add the parsed record to the data list
            data.append(rec)

        # Sort the records
        data.sort(key=lambda rec: rec["time"])

        # Return the data
        return data
