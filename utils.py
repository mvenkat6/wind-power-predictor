"""
Logging utilities
"""

__author__ = "Maitreya Venkataswamy"

import sys
import logging


def setup_logger(logger, output_file):
    # Set the logger level to INFO
    logger.setLevel(logging.INFO)

    # Set the STDOUT settings for the logger
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('[%(asctime)s]: %(message)s'))
    logger.addHandler(stdout_handler)

    # Set the file settings for the logger
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s]: %(message)s'))
    logger.addHandler(file_handler)
