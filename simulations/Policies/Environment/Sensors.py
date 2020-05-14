import numpy as nu
from PossibleActions import Action


class Sensor():
    def __init__(self, name, initValue):
        self.name = name
        self.value = float(initValue)


class SensorArray():
    def __init__(self, location):
        self.deltaEdgeWindSpeed = 0
        self.verticalAcceleration = 0
        self.v_r = 0
        self.v_lo = 0
        self.v_theta = 0
        self.bankAngle = 0
        self.pitchAngle = 0
        self.attackAngle = 0
        self.aileron = 0
        self.elevator = 0
        self.rudder = 0
        self.power = 0
        self.battery = 1
        self.latitude = location.latitude
        self.longitude = location.longitude
        self.altitude = 0
        self.pressure = 1  # bars


    def reset(self, location):
        self.deltaEdgeWindSpeed = 0
        self.verticalAcceleration = 0
        self.v_r = 0
        self.v_lo = 0
        self.v_theta = 0
        self.bankAngle = 0
        self.pitchAngle = 0
        self.attackAngle = 0
        self.aileron = 0
        self.elevator = 0
        self.rudder = 0
        self.power = 0
        self.battery = 1
        self.latitude = location.latitude
        self.longitude = location.longitude
        self.altitude = 0
        self.pressure = 1  # bars
        
        return nu.array([
            self.deltaEdgeWindSpeed,
            self.verticalAcceleration,
            self.v_r,
            self.v_lo,
            self.v_theta,
            self.bankAngle,
            self.pitchAngle,
            self.attackAngle,
            self.aileron,
            self.elevator,
            self.rudder,
            self.battery,
            self.power
        ])


    def conduct(self, action):
        # todo: set action to flight controls
        self.aileron += action.aileron
        self.elevator += action.elevator
        self.rudder += action.rudder
        self.power = action.power


    def readToState(self, virtualWeather):
        # update readings according to real or virtual weather condition

        return nu.array([
            self.deltaEdgeWindSpeed,
            self.verticalAcceleration,
            self.v_r,
            self.v_lo,
            self.v_theta,
            self.bankAngle,
            self.pitchAngle,
            self.attackAngle,
            self.aileron,
            self.elevator,
            self.rudder,
            self.battery,
            self.power
        ])

        return state
