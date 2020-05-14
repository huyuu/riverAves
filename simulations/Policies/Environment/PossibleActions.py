import numpy as nu


class Action():
    def __init__(self, aileronDelta, elevatorDelta, rudderDelta, power):
        self.aileronDelta = aileronDelta
        self.elevatorDelta = elevatorDelta
        self.rudderDelta = rudderDelta
        self.power = power


def getPossibleActions():
    aileronActions = [-0.6, -0.25, 0, 0.25, 0.6]
    elevatorActions = [-0.6, -0.25, 0, 0.25, 0.6]
    rudderActions = [-0.25, 0, 0.25]
    powerActions = [-1.0, 0, 0.5, 1.0]

    possibleActions = []
    for xIndex, aileronAction in enumerate(aileronActions):
        for yIndex, elevatorAction in enumerate(elevatorActions):
            for zIndex, rudderAction in enumerate(rudderActions):
                for wIndex, powerAction in enumerate(powerActions)
                possibleActions.append(Action(aileronAction, elevatorAction, rudderAction, powerAction))
    return possibleActions
