#!/bin/sh
# https://qiita.com/takayukioda/items/edf371b3566bea64d046


# set vars
# reference: https://qiita.com/rubytomato@github/items/173a812d7a8ec4646955#-算術式-
toFileName="$1"
latitudeLowerBound=`echo "$2" | bc`
latitudeUpperBound=`echo "$3" | bc`
longitudeLowerBound=`echo "$4" | bc`
longitudeUpperBound=`echo "$5" | bc`

touch weatherLog.txt
# translate it into csv file
# parameter reference: https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p50.f003.shtml
# /Users/yuyang/Downloads/grib2/wgrib2/wgrib2 "$toFileName.grb2" -match ":" -undefine out-box $longitudeLowerBound:$longitudeUpperBound $latitudeLowerBound:$latitudeUpperBound -csv "$toFileName.csv" >> weatherLog.txt
mkdir "$toFileName"

# process meter based data
for meter in 10 20 30 40 50 80 100
do
  # translate and compress East West Wind data
  /Users/yuyang/Downloads/grib2/wgrib2/wgrib2 "$toFileName.grb2" -match ":UGRD:$meter m above ground:" -undefine out-box $longitudeLowerBound:$longitudeUpperBound $latitudeLowerBound:$latitudeUpperBound -csv "$toFileName/${meter}_m_EWWind.csv" >> weatherLog.txt
  ex -s "$toFileName/${meter}_m_EWWind.csv" < weatherFixing.exscript
  # translate and compress North South Wind data
  /Users/yuyang/Downloads/grib2/wgrib2/wgrib2 "$toFileName.grb2" -match ":VGRD:$meter m above ground:" -undefine out-box $longitudeLowerBound:$longitudeUpperBound $latitudeLowerBound:$latitudeUpperBound -csv "$toFileName/${meter}_m_NSWind.csv" >> weatherLog.txt
  ex -s "$toFileName/${meter}_m_NSWind.csv" < weatherFixing.exscript
done

# process pressure based data
for mb in 1000 975 950 925 900 850 800 750 700 650 600 550 500 450 400 350 300 250 200
do
  # translate and compress East West Wind data
  /Users/yuyang/Downloads/grib2/wgrib2/wgrib2 "$toFileName.grb2" -match ":UGRD:$mb mb:" -undefine out-box $longitudeLowerBound:$longitudeUpperBound $latitudeLowerBound:$latitudeUpperBound -csv "$toFileName/${mb}_mb_EWWind.csv" >> weatherLog.txt
  ex -s "$toFileName/${mb}_mb_EWWind.csv" < weatherFixing.exscript
  # translate and compress North South Wind data
  /Users/yuyang/Downloads/grib2/wgrib2/wgrib2 "$toFileName.grb2" -match ":VGRD:$mb mb:" -undefine out-box $longitudeLowerBound:$longitudeUpperBound $latitudeLowerBound:$latitudeUpperBound -csv "$toFileName/${mb}_mb_NSWind.csv" >> weatherLog.txt
  ex -s "$toFileName/${mb}_mb_NSWind.csv" < weatherFixing.exscript
  # translate and compress Vertical Wind data
  /Users/yuyang/Downloads/grib2/wgrib2/wgrib2 "$toFileName.grb2" -match ":DZDT:$mb mb:" -undefine out-box $longitudeLowerBound:$longitudeUpperBound $latitudeLowerBound:$latitudeUpperBound -csv "$toFileName/${mb}_mb_VerticalWind.csv" >> weatherLog.txt
  ex -s "$toFileName/${mb}_mb_VerticalWind.csv" < weatherFixing.exscript
  # translate geopotential height
  /Users/yuyang/Downloads/grib2/wgrib2/wgrib2 "$toFileName.grb2" -match ":HGT:$mb mb:" -undefine out-box $longitudeLowerBound:$longitudeUpperBound $latitudeLowerBound:$latitudeUpperBound -csv "$toFileName/${mb}_mb_GeopotentialHeight.csv" >> weatherLog.txt
  ex -s "$toFileName/${mb}_mb_GeopotentialHeight.csv" < weatherFixing.exscript
done
