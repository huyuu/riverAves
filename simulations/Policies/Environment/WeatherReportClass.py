import numpy as nu
import pandas as pd
from scipy import interpolate
from sklearn import gaussian_process as gp
import pickle
import subprocess as sp
import multiprocessing as mp
import urllib.request as urlrequest
from urllib.error import URLError
import requests
import datetime as dt
import time
import os
import shutil
# self made modules
from LocationClass import Location, getGeoBoundsFromFlightPlan


class WeatherReport():
    def __init__(self, fromLocation, toLocation):
        # set geo location
        self.fromLocation = fromLocation
        self.toLocation = toLocation


    " get latest GFS report url "
    def getRecentWeatherReport(self):
        # get Geo bounds
        latitudeLowerBound, latitudeUpperBound, longitudeLowerBound, longitudeUpperBound = getGeoBoundsFromFlightPlan(self.fromLocation, self.toLocation)
        # init vars
        now = dt.datetime.utcnow()
        endTimeOfNeededForcast = now + dt.timedelta(days=2)
        referenceReleaseHour = 18  # should be one of 00, 06, 12, 18
        date = dt.datetime(now.year, now.month, now.day, referenceReleaseHour, 0, 0)
        baseurl = "https://nomads.ncdc.noaa.gov/data/gfs4"
        # find latest file on GFS
        print('Start searching for the latest weather forcast on GFS ...')
        while True:
            hourString = date.strftime("%H")
            yearMonthString = date.strftime("%Y%m")
            fullDateString = date.strftime("%Y%m%d")
            releaseTimeString = f'/{yearMonthString}/{fullDateString}/gfs_4_{fullDateString}_{referenceReleaseHour}00'
            lastComponent = "_000.grb2"
            url = baseurl + releaseTimeString + lastComponent

            # get available url
            response = requests.head(url)
            # reference: https://note.nkmk.me/python-requests-usage/
            if response.status_code == 200:
                print(f'GFS weather report found at {fullDateString}_1800.')
                break
            elif response.status_code == 404:
                # reference: https://note.nkmk.me/python-datetime-timedelta-measure-time/
                timeProgess = dt.timedelta(days=1)
                date -= timeProgess
            else:  # error didn't catch. wait 1 minute and try again
                print(f'Response with un expected error: status_code: {response.status_code}; header: {response.headers}. Try again after 1 minute...')
                time.sleep(60)

        # download and translate grib2 files
        print('Start downloading the latest weather reports from GFS ...')
        hoursFromReleaseTime = (now - date).total_seconds() / 3600
        hoursFromReleaseTime = (hoursFromReleaseTime // 3) * 3
        date += dt.timedelta(hours=hoursFromReleaseTime)
        processes = []
        # check if weatherForecasts dir exists
        if os.path.isdir("./weatherForecasts"):
            shutil.rmtree("./weatherForecasts")
            os.mkdir("./weatherForecasts")
        else:
            os.mkdir("./weatherForecasts")
        while date <= endTimeOfNeededForcast:
            storedName = "./weatherForecasts/" + date.strftime("%Y_%m_%d_%H")
            if hoursFromReleaseTime < 100:
                lastComponent = f'_0{int(hoursFromReleaseTime)}.grb2'
            else:
                lastComponent = f'_{int(hoursFromReleaseTime)}.grb2'
            url = baseurl + releaseTimeString + lastComponent
            process = mp.Process(target=downloadAndTranslateWeatherReportToCSVFile, args=(storedName, url, latitudeLowerBound, latitudeUpperBound, longitudeLowerBound, longitudeUpperBound))
            process.start()
            processes.append(process)
            # prepare for next loop
            hoursFromReleaseTime += 3
            date += dt.timedelta(hours=3)
        fileCount = len(processes)
        print(f'{fileCount} weather forecast files are in progress. This may take several minutes ...')
        # wait until all downloading processes are done
        for process in processes:
            process.join()

        # check results
        existingReportDirectories = len(os.listdir("./weatherForecasts"))
        if fileCount == existingReportDirectories:
            timeCostInMinutes = int((dt.datetime.utcnow() - now).total_seconds() // 60)
            print(f'All weather forecasts have been successfully downloaded and translated.　(time costs: {timeCostInMinutes} minutes)')
            os.remove("weatherLog.txt")
        else:
            print(f"Some files may be missing. Should have {fileCount} directories, but exactly {existingReportDirectories}.")


    # def computeInterpolationForSimulation(self):
    #     # spacial interpolation
    #     dirsOfTimeStamp = os.listdir('./weatherForecasts')
    #     processes = []
    #     for dir in dirsOfTimeStamp:
    #         process = mp.Process(target=computeSpacialInterpolation, args=(dir, ['VerticalWind', 'ESWind', 'NSWind']))
    #         process.start()
    #         processes.append(process)
    #     for process in processes:
    #         process.join()


    def getWindsAt(self, location, pressure, time):
        print(f'Start getting winds at {time}')
        # get file paths
        lowerTime = dt.datetime(time.year, time.month, time.day, (time.hour // 3)*3, 0, 0)
        upperTime = lowerTime + dt.timedelta(hours=3)
        timeDelta = (time - lowerTime).total_seconds() / 3600.0
        altitude = 1.0 / float(pressure)
        lowerPressure = floorInto(pressure, [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200])
        lowerAltitude = 1.0/float(lowerPressure)
        upperPressure = ceilInto(pressure, [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200])
        upperAltitude = 1.0/float(upperPressure)
        modelBaseName = lowerTime.strftime("%Y_%m_%d_%H") + "_" + upperTime.strftime("%Y_%m_%d_%H") + "_" + f'{lowerPressure}_{upperPressure}_mb'
        for parameter in ["NSWind"]:
            modelFileName = modelBaseName + f'_{parameter}_model.sav'
            if os.path.exists(modelFileName):
                print(f'Trained model {modelBaseName} found.')
                model = pickle.load(open(modelFileName, 'rb'))
                ll = pd.read_csv(lowerTimeLowerAltitudeFilePath, index_col=0)
                latitudes = ll.index.values.flatten()
                longitudes = ll.columns.values.astype(nu.float64).flatten()
            else:
                print(f'Trained model {modelBaseName} not found, start training ...')
                timeConsumptionStart = dt.datetime.now()
                lowerTimeLowerAltitudeFilePath = "./weatherForecasts/" + lowerTime.strftime("%Y_%m_%d_%H") + f'/{lowerPressure}_mb_{parameter}.csv'
                lowerTimeUpperAltitudeFilePath = "./weatherForecasts/" + lowerTime.strftime("%Y_%m_%d_%H") + f'/{upperPressure}_mb_{parameter}.csv'
                upperTimeLowerAltitudeFilePath = "./weatherForecasts/" + upperTime.strftime("%Y_%m_%d_%H") + f'/{lowerPressure}_mb_{parameter}.csv'
                upperTimeUpperAltitudeFilePath = "./weatherForecasts/" + upperTime.strftime("%Y_%m_%d_%H") + f'/{upperPressure}_mb_{parameter}.csv'
                ll = pd.read_csv(lowerTimeLowerAltitudeFilePath, index_col=0)
                lu = pd.read_csv(lowerTimeUpperAltitudeFilePath, index_col=0).values
                ul = pd.read_csv(upperTimeLowerAltitudeFilePath, index_col=0).values
                uu = pd.read_csv(upperTimeLowerAltitudeFilePath, index_col=0).values
                latitudes = ll.index.values.flatten()
                longitudes = ll.columns.values.astype(nu.float64).flatten()
                ll = ll.values
                trainLabels = nu.concatenate([ll.ravel(), lu.ravel(), ul.ravel(), uu.ravel()])
                del ll, lu, ul, uu
                # lowerLayer = (ul - ll) / (3.0) * timeDelta + ll
                # upperLayer = (uu - lu) / (3.0) * timeDelta + lu

                totalPoints = latitudes.shape[0]*longitudes.shape[0]
                _latitudes, _longitudes = nu.meshgrid(latitudes, longitudes, indexing='ij')
                # add features: latitude and longitude
                _ll = nu.concatenate([_latitudes.reshape(-1, 1), _longitudes.reshape(-1, 1)], axis=1)
                # add feature: altitude
                _ll = nu.concatenate([_ll, lowerAltitude * nu.ones((totalPoints, 1))], axis=1)
                # add feature: time
                _ll = nu.concatenate([_ll, 0 * nu.ones((totalPoints, 1))], axis=1)
                _lu = _ll
                _lu[:, 2] = upperAltitude
                _ul = _ll
                _ul[:, 3] = 3.0
                _uu = _ul
                _uu[:, 2] = upperAltitude
                trainSamples = nu.concatenate([_ll, _lu, _ul, _uu])
                del _ll, _lu, _ul, _uu

                testSamples = nu.array([[location.latitude, location.longitude , altitude, timeDelta]])

                # generate model
                # reference: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py
                kernel = 1.0 * gp.kernels.RBF(1.0)
                model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
                model.fit(trainSamples, trainLabels)
                pickle.dump(model, open(modelFileName, 'wb'))
                timeConsumption = (dt.datetime.now() - teimConsumptionStart).total_seconds() // 60
                print(f'Training ends. (time cost: {timeConsumption} minutes)')
            predictedLabels, std = model.predict(trainSamples[:totalPoints, :], return_std=True)
            data = pd.DataFrame(predictedLabels.reshape(latitudes.shape[0], longitudes.shape[0]), index=latitudes, columns=longitudes)
            data.to_csv(f'predicted_{parameter}.csv', header=True, index=True)


def computeInterpolation(lowerTimeLowerLocationFilePath, lowerTimeUpperLocationFilePath, upperTimeLowerLocationFilePath, upperTimeUpperLocationFilePath, lowerTime, upperTime, currentTime):
    # sp.call(['./computeInterpolation.jl', lowerTimeLowerLocationFilePath, lowerTimeUpperLocationFilePath, upperTimeLowerLocationFilePath, upperTimeUpperLocationFilePath])
    ll = pd.read_csv(lowerTimeLowerLocationFilePath, index_col=0)
    latitudes = ll.index.values
    longitudes = ll.columns.values.astype(nu.float64)
    ll = ll.values
    lu = pd.read_csv(lowerTimeUpperLocationFilePath, index_col=0).values
    ul = pd.read_csv(upperTimeLowerLocationFilePath, index_col=0).values
    uu = pd.read_csv(upperTimeUpperLocationFilePath, index_col=0).values

    lowerLayer = (ul - ll) / (upperTime - lowerTime) * (currentTime - lowerTime) + ll
    upperLayer = (uu - lu) / (upperTime - lowerTime) * (currentTime - lowerTime) + lu

    del ll, lu, ul, uu


    # csvfiles = os.listdir(f'./weatherForecasts/{dirName}')
    # for parameter in parameters:
    #     data = None
    #     for targetFile in filter(lambda name: parameter in name and (name.split('_')[1] == 'mb'), csvfiles):
    #         level, unit, _ = targetFile.split('_')
    #         level = int(level)




def downloadAndTranslateWeatherReportToCSVFile(storedName, url, latitudeLowerBound, latitudeUpperBound, longitudeLowerBound, longitudeUpperBound):
    # download grb2 file
    # https://qiita.com/orangain/items/0a641d980019fd7e0c52
    urlrequest.urlretrieve(url, f'{storedName}.grb2')
    # translate into csv file
    sp.call(['./getGFSWeatherReport.sh', f'{storedName}', f'{latitudeLowerBound}', f'{latitudeUpperBound}', f'{longitudeLowerBound}', f'{longitudeUpperBound}'])
    # remove grb2 file to save memory
    os.remove(f'{storedName}.grb2')
    # compressing data into grid form
    csvfiles = os.listdir(storedName)
    for csvfileName in csvfiles:
        path = f'{storedName}/{csvfileName}'
        data = pd.read_csv(path, header=None, names=['longitude', 'latitude', 'value'])
        latitudes = nu.sort(nu.unique(data['latitude'].values))
        longitudes = nu.sort(nu.unique(data['longitude'].values))
        data = data['value'].values.reshape([latitudes.shape[0], longitudes.shape[0]])
        # reference: https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#d-spline-representation-procedural-bisplrep
        # _latitudes, _longitudes = nu.meshgrid(latitudes, longitudes, indexing='ij')
        # interFunc = interpolate.bisplrep(_latitudes, _longitudes, data)
        # points = 50
        # newLatitudes = nu.linspace(latitudes[0], latitudes[-1], points)
        # newLongitudes = nu.linspace(longitudes[0], longitudes[-1], points)
        # data = interpolate.bisplev(newLatitudes, newLongitudes, interFunc)
        # newDataFrame = pd.DataFrame(data, index=newLatitudes, columns=newLongitudes)
        newDataFrame = pd.DataFrame(data, index=latitudes, columns=longitudes)
        os.remove(path)
        newDataFrame.to_csv(path, header=True, index=True)


def floorInto(origin, pressures):
    originAltitude = 1.0/float(origin)
    for index, pressure in enumerate(pressures):
        altitude = 1.0/float(pressure)
        if altitude > originAltitude:
            return pressures[index-1]
        else:
            continue
    return pressures[-1]


def ceilInto(origin, pressures):
    originAltitude = 1.0/float(origin)
    for index, pressure in enumerate(pressures):
        altitude = 1.0/float(pressure)
        if altitude >= originAltitude:
            return pressures[index]
        else:
            continue
    return pressures[-1]


if __name__ == '__main__':
    fromLocation = Location(latitude=38.2356225, longitude=140.8359262, altitude=100)
    toLocation = Location(latitude=22.9632839, longitude=120.2318360, altitude=0)
    weatherReport = WeatherReport(fromLocation=fromLocation, toLocation=toLocation)
    # weatherReport.getRecentWeatherReport()
    weatherReport.getWindsAt(fromLocation, 915, dt.datetime.utcnow())
