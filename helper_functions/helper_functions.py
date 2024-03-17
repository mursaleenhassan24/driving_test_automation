from constants import *


def calculate_pixel_per_meter(vehicle_width_pixels, vehicle_height_pixels):
    ACTUAL_VEHICLE_WIDTH_M = ACTUAL_VEHICLE_WIDTH_MM / 1000
    ACTUAL_VEHICLE_LENGTH_M = ACTUAL_VEHICLE_LENGTH_MM / 1000

    pixels_per_meter_width = vehicle_width_pixels / ACTUAL_VEHICLE_WIDTH_M
    pixels_per_meter_length = vehicle_height_pixels / ACTUAL_VEHICLE_LENGTH_M
    pixels_per_meter = (pixels_per_meter_width + pixels_per_meter_length) / 2
    return int(pixels_per_meter)