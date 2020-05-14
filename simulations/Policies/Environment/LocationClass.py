import numpy as nu


class Location():
    def __init__(self, latitude, longitude, altitude=0):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude


    def toNdarray(self):
        return nu.array([self.latitude, self.longitude])


    def __sub__(self, other):
        if isinstance(other, Location):
            return self.toNdarray() - other.toNdarray()
        elif isinstance(other, nu.ndarray):
            return self.toNdarray() - other
        else:
            raise TypeError


    def __add__(self, other):
        if isinstance(other, Location):
            return self.toNdarray() + other.toNdarray()
        elif isinstance(other, nu.ndarray):
            return self.toNdarray() + other
        else:
            raise TypeError


def getGeoBoundsFromFlightPlan(fromLocation, toLocation, scale=2):
    vector = toLocation - fromLocation
    # translate to unit vector
    vector = vector / nu.linalg.norm(vector, ord=2)
    # set latitude bounds
    if vector[0] <= 0:
        latitudeLowerBound = toLocation.latitude + vector[0] * scale
        latitudeUpperBound = fromLocation.latitude - vector[0] * scale
    else:
        latitudeLowerBound = fromLocation.latitude - vector[0] * scale
        latitudeUpperBound = toLocation.latitude + vector[0] * scale
    # set longitude bounds
    if vector[1] <= 0:
        longitudeLowerBound = toLocation.longitude + vector[1] * scale
        longitudeUpperBound = fromLocation.longitude - vector[1] * scale
    else:
        longitudeLowerBound = fromLocation.longitude - vector[1] * scale
        longitudeUpperBound = toLocation.longitude + vector[1] * scale
    return latitudeLowerBound, latitudeUpperBound, longitudeLowerBound, longitudeUpperBound
