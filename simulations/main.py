# reference: https://gitlab.com/zbalda/gym-pyfme/-/blob/master/gym_pyfme/envs/pyfme_env.py

import pyfme as pf
from pyfme.aircrafts import Cessna172
from pyfme.environment.wind import NoWind
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment.environment import Environment as PyfemEnviroment
from pyfme.utils.trimmer import steady_state_trim
from pyfme.models.state.position import GeodeticPosition


if __name__ == '__main__':
    # set aircraft
    aircraft = Cessna172()
    # set atmosphere
    atmosphere = ISA1976()
    # set wind
    wind = NoWind()
    # set gravity
    gravity = VerticalConstant()
    # set environment
    environment = PyfmeEnvironment(atmosphere, gravity, wind)

    # set initial values
    position = GeodeticPosition(lat=38, lon=140, height=10)
    yaw = 0  # rad
    airspeed = 1  # m/s
    controls = {'delta_elevator': 0, 'delta_aileron': 0, 'delta_rudder': 0, 'delta_t': 0.5}

    # calculate
    trimmed_state, trimmed_controls = steady_state_trim(
        aircraft,
        environment,
        position,
        yaw,
        airspeed,
        controls
    )
    print(trimmed_state)
