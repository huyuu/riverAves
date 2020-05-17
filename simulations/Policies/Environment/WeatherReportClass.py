import numpy as nu
import pandas as pd
from scipy import interpolate
from sklearn import gaussian_process as gp
import sklearn as sk
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


pressuresAvailable = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 1000]
parametersNeeded = ['NSWind', 'ESWind']


class WeatherReport():
    def __init__(self, fromLocation, toLocation):
        # set geo location
        self.fromLocation = fromLocation
        self.toLocation = toLocation
        self.currentModels = {}
        self.startTime = dt.datetime.utcnow()


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


    def generateNeighborModels(self, location, pressure, time, shouldRunOnBackground=True):
        if shouldRunOnBackground:
            process = mp.Process(target=_generateNeighborModelsProcess, args=(location, pressure, time, self.startTime))
            process.start()
            print('Start generating neighbor models in background.')
        else:
            print('Start generating neighbor models for weather prediction ... (This may take several minutes)')
            _start = dt.datetime.now()
            _generateNeighborModelsProcess(location, pressure, time, self.startTime)
            timeCost = (dt.datetime.now() - _start).total_seconds()
            print(f'Neighbor models generating completed. (time cost: {timeCost} seconds)')


    def getParametersAtPosition(self, location, pressure, time):
        modelDir = "./modelsForWeatherPrdicting"
        for parameter in parametersNeeded:
            # get low layer info from position
            lowLatitude = floorOf(location.latitude, latitudes)
            lowLatitudeIndex = nu.where(latitudes == lowLatitude)[0][0]
            lowLongitude = floorOf(location.longitude, longitudes)
            lowLongitudeIndex = nu.where(longitudes == lowLongitude)[0][0]
            highPressure = ceilOf(pressure, pressuresAvailable)
            lowTime = dt.datetime(time.year, time.month, time.day, (time.hour // 3)*3, 0, 0)
            lowTimeDirName = lowTime.strftime("%Y_%m_%d_%H")
            # get high layer info from position
            highLatitude = ceilOf(location.latitude, latitudes)
            highLatitudeIndex = nu.where(latitudes == highLatitude)[0][0]
            highLongitude = ceilOf(location.longitude, longitudes)
            highLongitudeIndex = nu.where(longitudes == highLongitude)[0][0]
            lowPressure = floorOf(pressure, pressuresAvailable)
            highTime = lowTime + dt.timedelta(hours=3)
            highTimeDirName = highTime.strftime("%Y_%m_%d_%H")
            # get model base name
            modelBaseName = f'{modelDir}/{lowLatitudeIndex}-{lowLatitude}-{highLatitudeIndex}-{highLatitude}-{lowLongitudeIndex}-{lowLongitude}-{highLongitudeIndex}-{highLongitude}-{lowTimeDirName}-{highTimeDirName}-{highPressure}-{lowPressure}-mb-{parameter}'
            modelFileName = f'{modelBaseName}.sav'
            # load stored model
            model = pickle.load(open(modelFileName, 'rb'))
            self.currentModels[parameter] = (model, modelBaseName)
            # predict label at position
            position = [location.latitude, location.longitude, 1.0/float(pressure), (time - self.startTime).total_seconds()]
            label = model(position[0], position[1], position[2], position[3])
            # add result to .csv file
            data = pd.read_csv(f'{modelBaseName}.csv', index_col=0)
            data.append({
                'latitude': position[0],
                'longitude': position[1],
                'altitude': position[2],
                'secondsFromStart': position[3],
                'value': label
            })
            data.to_csv(f'{modelBaseName}.csv', header=True, index=True)
            del data



    def getWindsAt(self, location, pressure, time):
        print(f'Start getting winds at {time}')

        # get file paths
        lowerTime = dt.datetime(time.year, time.month, time.day, (time.hour // 3)*3, 0, 0)
        upperTime = lowerTime + dt.timedelta(hours=3)
        timeDelta = (time - lowerTime).total_seconds() / 3600.0
        altitude = 1.0 / float(pressure)
        upperPressure = ceilOf(pressure, [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 1000])
        lowerAltitude = 1.0/float(upperPressure)
        lowerPressure = floorOf(pressure, [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 1000])
        upperAltitude = 1.0/float(lowerPressure)
        modelBaseName = lowerTime.strftime("%Y_%m_%d_%H") + "_" + upperTime.strftime("%m_%d_%H") + "_" + f'{upperPressure}_{lowerPressure}_mb'

        # search for models
        for parameter in ["NSWind"]:
            modelFileName = modelBaseName + f'_{parameter}_model.sav'
            if os.path.exists(modelFileName):
                print(f'Trained model {modelFileName} found.')
                model = pickle.load(open(modelFileName, 'rb'))
            else:
                print(f'Trained model {modelFileName} not found.')
                print('Start training ... (This may take several minutes)')
                timeConsumptionStart = dt.datetime.now()
                # load files
                lowerTimeLowerAltitudeFilePath = "./weatherForecasts/" + lowerTime.strftime("%Y_%m_%d_%H") + f'/{upperPressure}_mb_{parameter}.csv'
                lowerTimeUpperAltitudeFilePath = "./weatherForecasts/" + lowerTime.strftime("%Y_%m_%d_%H") + f'/{lowerPressure}_mb_{parameter}.csv'
                upperTimeLowerAltitudeFilePath = "./weatherForecasts/" + upperTime.strftime("%Y_%m_%d_%H") + f'/{upperPressure}_mb_{parameter}.csv'
                upperTimeUpperAltitudeFilePath = "./weatherForecasts/" + upperTime.strftime("%Y_%m_%d_%H") + f'/{lowerPressure}_mb_{parameter}.csv'
                ll = pd.read_csv(lowerTimeLowerAltitudeFilePath, index_col=0)
                lu = pd.read_csv(lowerTimeUpperAltitudeFilePath, index_col=0).values
                ul = pd.read_csv(upperTimeLowerAltitudeFilePath, index_col=0).values
                uu = pd.read_csv(upperTimeLowerAltitudeFilePath, index_col=0).values
                latitudes = ll.index.values.flatten()
                longitudes = ll.columns.values.astype(nu.float64).flatten()
                ll = ll.values

                trainLabels = nu.concatenate([ll.ravel(), lu.ravel(), ul.ravel(), uu.ravel()])
                # trainLabels = ll.ravel()
                del ll, lu, ul, uu
                # do preprocessing for training
                totalPoints = latitudes.shape[0]*longitudes.shape[0]
                _latitudes, _longitudes = nu.meshgrid(latitudes, longitudes, indexing='ij')
                # add features: latitude and longitude
                _ll = nu.concatenate([_latitudes.reshape(-1, 1), _longitudes.reshape(-1, 1)], axis=1)
                # add feature: altitude
                _ll = nu.concatenate([_ll, lowerAltitude * nu.ones((totalPoints, 1))], axis=1)
                # add feature: time
                _ll = nu.concatenate([_ll, 0 * nu.ones((totalPoints, 1))], axis=1)
                _lu = _ll.copy()
                _lu[:, 2] = upperAltitude
                _ul = _ll.copy()
                _ul[:, 3] = 3.0
                _uu = _ul.copy()
                _uu[:, 2] = upperAltitude
                # reference: https://note.nkmk.me/python-list-ndarray-dataframe-normalize-standardize/
                trainSamples = nu.concatenate([_ll, _lu, _ul, _uu])
                testSamples = _ll.copy()
                testSamples[:, 2] = lowerAltitude
                testSamples[:, 3] = timeDelta

                del _ll, _lu, _ul, _uu
                # testSamples = nu.array([[location.latitude, location.longitude , altitude, timeDelta]])

                model = interpolate.Rbf(trainSamples[:, 0], trainSamples[:, 1], trainSamples[:, 2], trainSamples[:, 3], trainLabels, function='multiquadric')
                # reference: https://localab.jp/blog/save-and-load-machine-learning-models-in-python-with-scikit-learn/
                pickle.dump(model, open(modelFileName, 'wb'))
                timeConsumption = (dt.datetime.now() - timeConsumptionStart).total_seconds() // 60
                print(f'Training ends. (time cost: {timeConsumption} minutes)')
            self.model = model
            predictedLabels = model(testSamples[:, 0], testSamples[:, 1], testSamples[:, 2], testSamples[:, 3])
            data = pd.DataFrame(predictedLabels.reshape(latitudes.shape[0], longitudes.shape[0]), index=latitudes, columns=longitudes)
            data.to_csv(f'predicted_{parameter}.csv', header=True, index=True)
            # if os.path.exists(modelFileName):
            #     print(f'Trained model {modelFileName} found.')
            #     model = pickle.load(open(modelFileName, 'rb'))
            # else:
            #     print(f'Trained model {modelFileName} not found.')
            #     print('Start training ... (This may take several minutes)')
            #     timeConsumptionStart = dt.datetime.now()
            #     # generate model
            #     # reference: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py
            #     longtermKernel = 1.0 * gp.kernels.RBF(length_scale=10)
            #     shortermKernel = 1.0 * gp.kernels.RationalQuadratic(length_scale=0.5)
            #     kernel = longtermKernel + shortermKernel
            #     model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            #     model.fit(trainSamples, trainLabels)
            #     # reference: https://localab.jp/blog/save-and-load-machine-learning-models-in-python-with-scikit-learn/
            #     pickle.dump(model, open(modelFileName, 'wb'))
            #     timeConsumption = (dt.datetime.now() - timeConsumptionStart).total_seconds() // 60
            #     print(f'Training ends. (time cost: {timeConsumption} minutes)')
            #
            # print(f'Model: {model.kernel_}')
            # print(f'Log_Marginal_Likelihood: {model.log_marginal_likelihood(model.kernel_.theta)}')
            # self.model = model
            # predictedLabels, std = model.predict(testSamples, return_std=True)
            # data = pd.DataFrame(predictedLabels.reshape(latitudes.shape[0], longitudes.shape[0]), index=latitudes, columns=longitudes)
            # data.to_csv(f'predicted_{parameter}.csv', header=True, index=True)


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


def floorOf(value, container):
    container.sort()
    for index, element in enumerate(container):
        if element >= value:
            return container[index-1]
    return container[-1]


def ceilOf(value, container):
    container.sort()
    for index, element in enumerate(container):
        if element > value:
            return element
    return container[-1]


def _generateNeighborModelsProcess(location, pressure, time, startTime):
    # read latitudes and longitudes from random csv file
    firstDir = os.listdir("./weatherForecasts")[3]
    firstFilePath = os.listdir(f'./weatherForecasts/{firstDir}/')[0]
    data = pd.read_csv(f'./weatherForecasts/{firstDir}/{firstFilePath}', index_col=0)
    latitudes = data.index.values.flatten()
    longitudes = data.columns.values.astype(nu.float64).flatten()
    del data
    latitudeInterval = nu.abs(latitudes[0] - latitudes[1])
    longitudeInterval = nu.abs(longitudes[0] - longitudes[1])

    modelDir = "./modelsForWeatherPrdicting"
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)

    # get neighbor latitudes
    neighborLatitudes = []
    for la in [location.latitude - latitudeInterval, location.latitude, location.latitude + latitudeInterval]:
        if nu.min(latitudes) <= la < nu.max(latitudes):
            neighborLatitudes.append(la)
    # get neighbor longitudes
    neighborLongitudes = []
    for lo in [location.longitude - longitudeInterval, location.longitude, location.longitude + longitudeInterval]:
        if nu.min(longitudes) <= lo < nu.max(longitudes):
            neighborLongitudes.append(lo)
    # get neighbor pressures
    highPressure = ceilOf(pressure, pressuresAvailable)
    lowPressure = ceilOf(pressure, pressuresAvailable)
    neighborPressures = [ highPressure - 10, pressure, lowPressure + 10 ]
    # get neighbor time
    neighborTimes = [time, time + dt.timedelta(hours=3)]
    # main calculate
    processes = []
    for la in neighborLatitudes:
        for lo in neighborLongitudes:
            for pre in neighborPressures:
                for t in neighborTimes:
                    for parameter in parametersNeeded:
                        # get low layer info from position
                        lowLatitude = floorOf(la, latitudes)
                        lowLatitudeIndex = nu.where(latitudes == lowLatitude)[0][0]
                        lowLongitude = floorOf(lo, longitudes)
                        lowLongitudeIndex = nu.where(longitudes == lowLongitude)[0][0]
                        highPressure = ceilOf(pre, pressuresAvailable)
                        lowTime = dt.datetime(t.year, t.month, t.day, (t.hour // 3)*3, 0, 0)
                        lowTimeDirName = lowTime.strftime("%Y_%m_%d_%H")
                        # get high layer info from position
                        highLatitude = ceilOf(la, latitudes)
                        highLatitudeIndex = nu.where(latitudes == highLatitude)[0][0]
                        highLongitude = ceilOf(lo, longitudes)
                        highLongitudeIndex = nu.where(longitudes == highLongitude)[0][0]
                        lowPressure = floorOf(pre, pressuresAvailable)
                        highTime = lowTime + dt.timedelta(hours=3)
                        highTimeDirName = highTime.strftime("%Y_%m_%d_%H")
                        # get model base name
                        modelBaseName = f'{modelDir}/{lowLatitudeIndex}-{lowLatitude}-{highLatitudeIndex}-{highLatitude}-{lowLongitudeIndex}-{lowLongitude}-{highLongitudeIndex}-{highLongitude}-{lowTimeDirName}-{highTimeDirName}-{highPressure}-{lowPressure}-mb-{parameter}'
                        if os.path.exists(f'{modelBaseName}.sav') and os.path.exists(f'{modelBaseName}.csv'):
                            updateWeatherModelWithSamples(modelBaseName)
                        else:
                            createWeatherModel(modelBaseName, startTime)


def createWeatherModel(modelBaseName, startTime):
    lowLatitudeIndex, lowLatitude, highLatitudeIndex, highLatitude, lowLongitudeIndex, lowLongitude, highLongitudeIndex, highLongitude, lowTimeDirName, highTimeDirName, highPressure, lowPressure, _, parameter = modelBaseName.split('/')[2].split('-')
    trainSamples = []
    trainLabels = []
    for timeDirName in [lowTimeDirName, highTimeDirName]:
        yearString, monthString, dayString, hourString = timeDirName.split('_')
        time = dt.datetime(int(yearString), int(monthString), int(dayString), int(hourString), 0, 0)
        secondsFromStart = (time - startTime).total_seconds()
        for pressure in [highPressure, lowPressure]:
            altitude = 1.0 / float(pressure)
            filePath = "./weatherForecasts/" + timeDirName + f'/{pressure}_mb_{parameter}.csv'
            for latitudeIndex, latitude in [(lowLatitudeIndex, lowLatitude), (highLatitudeIndex, highLatitude)]:
                for longitudeIndex, longitude in [(lowLongitudeIndex, lowLongitude), (highLongitudeIndex, highLongitude)]:
                    trainSamples.append([latitude, longitude, altitude, secondsFromStart])
                    valueAtPoint = pd.read_csv(filePath, index_col=0).iloc[int(latitudeIndex), int(longitudeIndex)]
                    trainLabels.append(valueAtPoint)
    trainSamples = nu.array(trainSamples)
    trainLabels = nu.array(trainLabels)
    # save training points
    trainingData = pd.DataFrame(nu.concatenate([trainSamples, trainLabels.reshape(-1, 1)], axis=1), columns=['latitude', 'longitude', 'altitude', 'secondsFromStart', 'value'])
    trainingData.to_csv(modelBaseName + ".csv", header=True, index=True)
    del trainingData
    # start training
    model = interpolate.Rbf(trainSamples[:, 0], trainSamples[:, 1], trainSamples[:, 2], trainSamples[:, 3], trainLabels, function='multiquadric')
    # reference: https://localab.jp/blog/save-and-load-machine-learning-models-in-python-with-scikit-learn/
    modelFileName = modelBaseName + ".sav"
    pickle.dump(model, open(modelFileName, 'wb'))


def updateWeatherModelWithSamples(modelBaseName, newPoints=None):
    pointsFilePath = modelBaseName + ".csv"
    storedPoints = pd.read_csv(pointsFilePath, index_col=0).values
    # if newPoints is applied, add them to trainingSamples
    if newPoints:
        updatedPoints = nu.concatenate([storedPoints, newPoints])
        del storedPoints
        trainSamples = updatedPoints[:, :4]
        trainLabels = updatedPoints[:, 4]
    else:  # if newPoint is None, new points must be stored in the {modelBaseName}.csv file.
        trainSamples = storedPoints[:, :4]
        trainLabels = storedPoints[:, 4]

    # start training
    model = interpolate.Rbf(trainSamples[:, 0], trainSamples[:, 1], trainSamples[:, 2], trainSamples[:, 3], trainLabels, function='multiquadric')
    # reference: https://localab.jp/blog/save-and-load-machine-learning-models-in-python-with-scikit-learn/
    modelFileName = modelBaseName + ".sav"
    pickle.dump(model, open(modelFileName, 'wb'))



if __name__ == '__main__':
    fromLocation = Location(latitude=38.2356225, longitude=140.8359262, altitude=100)
    toLocation = Location(latitude=22.9632839, longitude=120.2318360, altitude=0)
    weatherReport = WeatherReport(fromLocation=fromLocation, toLocation=toLocation)
    # weatherReport.getRecentWeatherReport()
    time = dt.datetime(2020, 5, 14, 10, 0, 0)
    # weatherReport.generateNeighborModels(fromLocation, 915, time, shouldRunOnBackground=True)
    weatherReport.getParametersAtPosition(fromLocation, 915, time)
    # weatherReport.getWindsAt(fromLocation, 915, time)
