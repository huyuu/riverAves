# Pyfme models
from pyfme.aircrafts import Cessna172
from pyfme.environment.wind import NoWind
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.environment import Environment as PyfemEnviroment
from pyfme.utils.trimmer import steady_state_trim
from pyfme.models.state.position import GeodeticPosition
from pyfme.models import EulerFlatEarth
# Foundations
import numpy as nu
import math as ma
import datetime as dt
# Custom modules
from possibleActions import getPossibleActions
from Sensors import SensorArray


class Environment():
    def __init__(self):
        # set environment
        atmosphere = ISA1976()
        wind = NoWind()
        gravity = VerticalConstant()
        self.pyfmeEnv = PyfmeEnvironment(atmosphere, gravity, wind)
        # set actions; aileron, elevator, battery
        self.possibleActions = getPossibleActions()
        # set states
        self.sensorArray = SensorArray()
        # set target pressure
        self.targetPressure = 0.5  # bars
        # set start utc time
        self.startTime = dt.datetime.utcnow()
        # set predicted consuming time
        self.predictedConsumingTime = 3600 * 2  # 2 hours


    def reset(self):
        self.sensorArray.reset()
        return self.sensorArray


    " Returns newState, didEnd, reward "
    def step(self, action):
        self.sensorArray.conduct(action)
        didEnd, reward = self.__checkIfDidEnd()
        return self.sensorArray.readToState(weather), didEnd, reward


    def __checkIfDidEnd(self):
        if self.sensorArray.bankAngle > ma.radians(30):
            return True, -1
        elif self.sensorArray.altitude < 10:
            return True, -1
        elif self.sensorArray.battery < 0.1:
            return True, -1
        elif self.sensorArray.pressure >= self.targetPressure:
            endTime = dt.datetime.utcnow()
            timeReward = self.predictedConsumingTime/((endTime - self.startTime).total_seconds())
            return True, 1 + timeReward
        else:
            return False, 0
