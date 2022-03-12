#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################      INSTRUCTIONS    FOR     USERS       ##########################
#####       0.) Input data should have shape (timestamps, levels, latitude, longitude), #####
#####           NetCDF4 format, daily means should be grouped into separate single      #####
#####           years, i.e. len(timestamps)=365 or 366 (leap years)                     #####
#####       1.) Set the parameters in lines 38 to 64 valid for your data.               #####
#####       2.) Set the name of your data in lines 72 and 73.                           #####
#####       3.) In lines 142 to 160 change "/path/to/data/ and set path to your data.   #####
#####       4.) In lines 164 to 187 check if variables in your netcdf data have the same
#####           name as in this code - if not, set the right names of variables.        #####
#####       5.) If you have data from reanalyses other than ERA5 and/or ERA-Interim, change
#####           all expressions "ERA5" and/or "ERAI" in the code with adequate ones.    #####
#####       6.) If start year of your data is different than 1950 and/or 1979, change all
#####           expressions "1950" and/or "1979" in the code with adequate ones.        #####

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
#import urllib
from netCDF4 import Dataset as NetCDFFile
from scipy.io import netcdf
from scipy import stats
import scipy.sparse
import scipy.sparse.linalg
#import pymannkendall as mk
#from mpl_toolkits.basemap import Basemap
#from statistics import mean
from scipy import integrate
import time
from time import ctime
import os
import gc

class KuoEliassen():
    N_x = 360                       # number of grid points in zonal direction
    N_y = 181                       # number of grid points in meridional direction
    N_border_lat = 40               # border latitude index for area from 50° to -50° (for drawings)
    N_pressure = 37                 # number of pressure layers
    N_period = 69                   # number of years in time period 1950-2018
    N_year = 365                    # days in a year
    N_leap_year = 366               # days in leap year
    N_jan = 31                      # cumulative days from 1st of January to 31st of January
    N_feb = 59                      # cumulative days from 1st of January to 28st of February
    N_mar = 90                      # cumulative days from 1st of January to 31st of March
    N_apr = 120                     # cumulative days from 1st of January to 30st of April
    N_may = 151                     # cumulative days from 1st of January to 31st of May
    N_jun = 181                     # cumulative days from 1st of January to 30st of June
    N_jul = 212                     # cumulative days from 1st of January to 31st of July
    N_aug = 243                     # cumulative days from 1st of January to 31st of August
    N_sep = 273                     # cumulative days from 1st of January to 30st of September
    N_oct = 304                     # cumulative days from 1st of January to 31st of October
    N_nov = 334                     # cumulative days from 1st of January to 30st of November
    N_dec = 365                     # cumulative days from 1st of January to 31st of December
    R_Earth = 6371000.              # Earth radius in meters
    g = 9.81                        # gravitational acceleration in m/s^2
    R_d = 287.                      # gas constant for dry air in J/kgK
    c_p = 1004.                     # specific heat at constant pressure in J/kgK
    p_0 = 1000.                     # surface pressure constant; needed in PDE for theta (potential temperature)
    N_iterations = 75               # number of iterations for successive overrelaxation (SOR) method for solving elliptic problem
    N_x_global_west = 0             # longitude -180°
    N_x_global_east = 359           # longitude 179°

    def __init__(self, reanalysis, area, period, start_year):
        self.reanalysis = reanalysis
        self.area = area
        self.period = period
        self.start_year = start_year
        
        self.data_ERA5 = ["era5_{:4d}.nc".format(i) for i in range(1950, 2019)]                    # NetCDF file with 1°x1° resolution data
        self.data_ERAI = ["erai_{:4d}.nc".format(i) for i in range(1979, 2019)]                # NetCDF file with 1°x1° resolution data

        return

    def computation(self):
        if self.reanalysis == "ERA5" and self.start_year == "1950":
            N_period = self.N_period
            N_pressure = self.N_pressure
            N_y = self.N_y
            N_x = self.N_x
            N_border_lat = self.N_border_lat
            start_period = 1950
            end_period = 2018
        elif self.reanalysis == "ERAI" or (self.reanalysis == "ERA5" and self.start_year == "1979"):
            N_period = self.N_period - 29
            N_pressure = self.N_pressure
            N_y = self.N_y
            N_x = self.N_x
            N_border_lat = self.N_border_lat
            start_period = 1979
            end_period = 2018

        dim = N_pressure*N_y

        lon = np.zeros((N_x), dtype=np.float32)
        lat = np.zeros((N_y), dtype=np.float32)
        pressure = np.zeros((N_pressure), dtype=np.float32)
        year = np.zeros((N_period), dtype=np.float32)
        
        radian_constant = np.pi/180.
        Cor_param = ((2*2*np.pi)/(24*3600))

        meridion_momentum_flux_dec = np.zeros((2, 31, N_pressure, N_y, N_x), dtype=np.float32)
        meridion_heat_flux_dec = np.zeros((2, 31, N_pressure, N_y, N_x), dtype=np.float32)
        omega_flux_dec = np.zeros((2, 31, N_pressure, N_y, N_x), dtype=np.float32)
        theta_flux_dec = np.zeros((2, 31, N_pressure, N_y, N_x), dtype=np.float32)
        diabatic_heating_zonally_dec = np.zeros((2, 31, N_pressure, N_y), dtype=np.float32)
            
        diabatic_heating_averaged_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        meridion_heat_flux_averaged_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        friction_averaged_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        meridion_momentum_flux_averaged_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        omega_flux_averaged_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        theta_flux_averaged_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        streamfunction_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        #streamfunction_SOR_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        diabatic_heating_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        meridion_heat_flux_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        friction_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        momentum_flux_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        omega_flux_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        theta_flux_yearly = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        linear_operator_L_yearly = np.zeros((N_period, dim, dim), dtype=np.float32)
        linear_operator_L_S2_yearly = np.zeros((N_period, dim, dim), dtype=np.float32)
        linear_operator_L_u_yearly = np.zeros((N_period, dim, dim), dtype=np.float32)
        linear_operator_L_T_yearly = np.zeros((N_period, dim, dim), dtype=np.float32)
        vector_psi_yearly = np.zeros((N_period, dim), dtype=np.float32)
        vector_D_yearly = np.zeros((N_period, dim), dtype=np.float32)
        psi_J = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        psi_vT = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        psi_Fx = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        psi_uv = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        psi_uw = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
        psi_wTh = np.zeros((N_period, N_pressure, N_y), dtype=np.float32)
            
        m = 0
        while m < N_period:
            if self.reanalysis == "ERA5" and self.start_year == "1950":
                year[m] = (1950 + m)
                data = NetCDFFile("/path/to/data/"+self.data_ERA5[m], "r")
                if m > 0:
                    data_previous_year = NetCDFFile("/path/to/data/"+self.data_ERA5[m-1], "r")
                if m < (N_period - 1):
                    data_next_year = NetCDFFile("/path/to/data/"+self.data_ERA5[m+1], "r")
            elif self.reanalysis == "ERA5" and self.start_year == "1979":
                year[m] = (1979 + m)
                data = NetCDFFile("/path/to/data/"+self.data_ERA5[m+29], "r")
                if m > 0:
                    data_previous_year = NetCDFFile("/path/to/data/"+self.data_ERA5[m+29-1], "r")
                if m < (N_period - 1):
                    data_next_year = NetCDFFile("/path/to/data/"+self.data_ERA5[m+29+1], "r")
            elif self.reanalysis == "ERAI":
                year[m] = (1979 + m)
                data = NetCDFFile("/path/to/data/"+self.data_ERAI[m], "r")
                if m > 0:
                    data_previous_year = NetCDFFile("/path/to/data/"+self.data_ERAI[m-1], "r")
                if m < (N_period - 1):
                    data_next_year = NetCDFFile("/path/to/data/"+self.data_ERAI[m+1], "r")

            print(year)

            zon_wind = data.variables['u'][:,:,:,:]              # zonal wind for each day in whole 3D grid
            mer_wind = data.variables['v'][:,:,:,:]              # meridional wind for each day in whole 3D grid
            ver_wind = data.variables['w'][:,:,:,:]              # vertical velocity for each day in whole 3D grid
            temp = data.variables['t'][:,:,:,:]                  # temperature for each day in whole 3D grid
            #time = data.variables['time'][:]                       # hours up from 1.1.1900

            if m > 0:
                zon_wind_previous = data_previous_year.variables['u'][-31:,:,:,:]
                mer_wind_previous = data_previous_year.variables['v'][-31:,:,:,:]
                ver_wind_previous = data_previous_year.variables['w'][-31:,:,:,:]
                temp_previous = data_previous_year.variables['t'][-31:,:,:,:]
            
            if m < (N_period - 1):
                zon_wind_next = data_next_year.variables['u'][0:15,:,:,:]
                mer_wind_next = data_next_year.variables['v'][0:15,:,:,:]
                ver_wind_next = data_next_year.variables['w'][0:15,:,:,:]
                temp_next = data_next_year.variables['t'][0:15,:,:,:]
            
            if m == 0:
                _lon_ = data.dimensions['longitude']                # 360
                _lat_ = data.dimensions['latitude']                 # 181
                _lon_ = data.variables['longitude'][:]              # longitude from 0° to 359°
                _lat_ = data.variables['latitude'][:]               # latitude from 90° to -90°
                level = data.variables['level'][:]                  # pressure level in mbar
                
                # writing pressure, lon and lat values in right order
                pressure[:] = level[::-1]
                lat[:] = _lat_[::-1]
                lon[0:int(N_x/2)] = np.subtract(_lon_[int(N_x/2):], 360)
                lon[int(N_x/2):] = _lon_[0:int(N_x/2)]
            
                print(pressure)
                print(lat)
                print(lon)
                
                np.save("pressure_KE", pressure)
                np.save("latitude_KE", lat)
                np.save("longitude_KE", lon)

                del _lon_, _lat_, level

            if year[m] % 4 == 0 and (year[m] % 100 != 0 or year[m] % 400 == 0):
                N_year = self.N_leap_year
                N_jan = 31
                N_feb = 60
                N_mar = 91
                N_apr = 121
                N_may = 152
                N_jun = 182
                N_jul = 213
                N_aug = 244
                N_sep = 274
                N_oct = 305
                N_nov = 335
            else:
                N_year = self.N_year
                N_jan = 31
                N_feb = 59
                N_mar = 90
                N_apr = 120
                N_may = 151
                N_jun = 181
                N_jul = 212
                N_aug = 243
                N_sep = 273
                N_oct = 304
                N_nov = 334

            # writing all the values by days from 1950/1979 to 2018
            zonal_wind = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            zonal_wind[:, :, :, 0:int(N_x/2)] = zon_wind[:, ::-1, ::-1, int(N_x/2):]
            zonal_wind[:, :, :, int(N_x/2):] = zon_wind[:, ::-1, ::-1, 0:int(N_x/2)]
            meridional_wind = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            meridional_wind[:, :, :, 0:int(N_x/2)] = mer_wind[:, ::-1, ::-1, int(N_x/2):]
            meridional_wind[:, :, :, int(N_x/2):] = mer_wind[:, ::-1, ::-1, 0:int(N_x/2)]
            omega = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            omega[:, :, :, 0:int(N_x/2)] = ver_wind[:, ::-1, ::-1, int(N_x/2):]
            omega[:, :, :, int(N_x/2):] = ver_wind[:, ::-1, ::-1, 0:int(N_x/2)]
            temperature = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            temperature[:, :, :, 0:int(N_x/2)] = temp[:, ::-1, ::-1, int(N_x/2):]
            temperature[:, :, :, int(N_x/2):] = temp[:, ::-1, ::-1, 0:int(N_x/2)]
            theta = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            k = 0
            while k < N_pressure:
                theta[:, k, :, :] = temperature[:, k, :, :]*((self.p_0/pressure[k])**(self.R_d/self.c_p))
                k = k + 1

            if m > 0:
                zonal_wind_previous = np.zeros((31, N_pressure, N_y, N_x), dtype=np.float32)
                zonal_wind_previous[:, :, :, 0:int(N_x/2)] = zon_wind_previous[:, ::-1, ::-1, int(N_x/2):]
                zonal_wind_previous[:, :, :, int(N_x/2):] = zon_wind_previous[:, ::-1, ::-1, 0:int(N_x/2)]
                meridional_wind_previous = np.zeros((31, N_pressure, N_y, N_x), dtype=np.float32)
                meridional_wind_previous[:, :, :, 0:int(N_x/2)] = mer_wind_previous[:, ::-1, ::-1, int(N_x/2):]
                meridional_wind_previous[:, :, :, int(N_x/2):] = mer_wind_previous[:, ::-1, ::-1, 0:int(N_x/2)]
                omega_previous = np.zeros((31, N_pressure, N_y, N_x), dtype=np.float32)
                omega_previous[:, :, :, 0:int(N_x/2)] = ver_wind_previous[:, ::-1, ::-1, int(N_x/2):]
                omega_previous[:, :, :, int(N_x/2):] = ver_wind_previous[:, ::-1, ::-1, 0:int(N_x/2)]
                temperature_previous = np.zeros((31, N_pressure, N_y, N_x), dtype=np.float32)
                temperature_previous[:, :, :, 0:int(N_x/2)] = temp_previous[:, ::-1, ::-1, int(N_x/2):]
                temperature_previous[:, :, :, int(N_x/2):] = temp_previous[:, ::-1, ::-1, 0:int(N_x/2)]
                theta_previous = np.zeros((31, N_pressure, N_y, N_x), dtype=np.float32)
                k = 0
                while k < N_pressure:
                    theta_previous[:, k, :, :] = temperature_previous[:, k, :, :]*((self.p_0/pressure[k])**(self.R_d/self.c_p))
                    k = k + 1
                #theta_previous[:, :, :, :] = temperature_previous[:, :, :, :]*((self.p_0/pressure[:])**(self.R_d/self.c_p))
                
            if m < (N_period - 1):
                zonal_wind_next = np.zeros((15, N_pressure, N_y, N_x), dtype=np.float32)
                zonal_wind_next[:, :, :, 0:int(N_x/2)] = zon_wind_next[:, ::-1, ::-1, int(N_x/2):]
                zonal_wind_next[:, :, :, int(N_x/2):] = zon_wind_next[:, ::-1, ::-1, 0:int(N_x/2)]
                meridional_wind_next = np.zeros((15, N_pressure, N_y, N_x), dtype=np.float32)
                meridional_wind_next[:, :, :, 0:int(N_x/2)] = mer_wind_next[:, ::-1, ::-1, int(N_x/2):]
                meridional_wind_next[:, :, :, int(N_x/2):] = mer_wind_next[:, ::-1, ::-1, 0:int(N_x/2)]
                omega_next = np.zeros((15, N_pressure, N_y, N_x), dtype=np.float32)
                omega_next[:, :, :, 0:int(N_x/2)] = ver_wind_next[:, ::-1, ::-1, int(N_x/2):]
                omega_next[:, :, :, int(N_x/2):] = ver_wind_next[:, ::-1, ::-1, 0:int(N_x/2)]
                temperature_next = np.zeros((15, N_pressure, N_y, N_x), dtype=np.float32)
                temperature_next[:, :, :, 0:int(N_x/2)] = temp_next[:, ::-1, ::-1, int(N_x/2):]
                temperature_next[:, :, :, int(N_x/2):] = temp_next[:, ::-1, ::-1, 0:int(N_x/2)]
                theta_next = np.zeros((15, N_pressure, N_y, N_x), dtype=np.float32)
                k = 0
                while k < N_pressure:
                    theta_next[:, k, :, :] = temperature_next[:, k, :, :]*((self.p_0/pressure[k])**(self.R_d/self.c_p))
                    k = k + 1
                #theta_next[:, :, :, :] = temperature_next[:, :, :, :]*((self.p_0/pressure[:])**(self.R_d/self.c_p))
                    
            print("wind and temperature data, time: "+ctime())
                
            print(len(zonal_wind), len(zonal_wind[0]), len(zonal_wind[0][0]), len(zonal_wind[0][0][0]))
            print(zonal_wind[10, 0, 0, 0], zon_wind[10, 36, 180, 180])
                
            print(len(meridional_wind), len(meridional_wind[0]), len(meridional_wind[0][0]), len(meridional_wind[0][0][0]))
            print(meridional_wind[10, 0, 0, 0], mer_wind[10, 36, 180, 180])
            
            print(len(omega), len(omega[0]), len(omega[0][0]), len(omega[0][0][0]))
            print(omega[10, 0, 0, 0], ver_wind[10, 36, 180, 180])
            
            print(len(temperature), len(temperature[0]), len(temperature[0][0]), len(temperature[0][0][0]))
            print(temperature[10, 0, 0, 0], temp[10, 36, 180, 180])
            
            print(len(theta), len(theta[0]), len(theta[0][0]), len(theta[0][0][0]))
            print(theta[10, 10, 10, 10], temperature[10, 10, 10, 10]*((self.p_0/pressure[10])**(self.R_d/self.c_p)))
            
            del zon_wind, mer_wind, ver_wind, temp
            gc.collect()
            data.close()
            if m > 0:
                del zon_wind_previous, mer_wind_previous, ver_wind_previous, temp_previous
                data_previous_year.close()
            if m < (N_period - 1):
                del zon_wind_next, mer_wind_next, ver_wind_next, temp_next
                data_next_year.close()

            if m > 0:
                print(len(zonal_wind_previous), len(meridional_wind_previous), len(omega_previous), len(temperature_previous), len(theta_previous))
            if m < (N_period - 1):
                print(len(zonal_wind_next), len(meridional_wind_next), len(omega_next), len(temperature_next), len(theta_next))

            u_moving_average = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            v_moving_average = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            omega_moving_average = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            T_moving_average = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            theta_moving_average = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            n = 0
            while n < N_year:
                if m == 0 and n < 15:
                    u_moving_average[n, :, :, :] = np.mean(zonal_wind[0:16+n, :, :, :], axis=0)
                    v_moving_average[n, :, :, :] = np.mean(meridional_wind[0:16+n, :, :, :], axis=0)
                    omega_moving_average[n, :, :, :] = np.mean(omega[0:16+n, :, :, :], axis=0)
                    T_moving_average[n, :, :, :] = np.mean(temperature[0:16+n, :, :, :], axis=0)
                    theta_moving_average[n, :, :, :] = np.mean(theta[0:16+n, :, :, :], axis=0)
                    n = n + 1
                if m == (N_period - 1) and n > (N_year - 16):
                    u_moving_average[n, :, :, :] = np.mean(zonal_wind[n-15:N_year, :, :, :], axis=0)
                    v_moving_average[n, :, :, :] = np.mean(meridional_wind[n-15:N_year, :, :, :], axis=0)
                    omega_moving_average[n, :, :, :] = np.mean(omega[n-15:N_year, :, :, :], axis=0)
                    T_moving_average[n, :, :, :] = np.mean(temperature[n-15:N_year, :, :, :], axis=0)
                    theta_moving_average[n, :, :, :] = np.mean(theta[n-15:N_year, :, :, :], axis=0)
                    n = n + 1
                if m > 0 and m < N_period and n < 15:
                    u_moving_average[n, :, :, :] = (np.sum(zonal_wind[0:16+n, :, :, :], axis=0) + np.sum(zonal_wind_previous[16+n:, :, :, :], axis=0))/31
                    v_moving_average[n, :, :, :] = (np.sum(meridional_wind[0:16+n, :, :, :], axis=0) + np.sum(meridional_wind_previous[16+n:, :, :, :], axis=0))/31
                    omega_moving_average[n, :, :, :] = (np.sum(omega[0:16+n, :, :, :], axis=0) + np.sum(omega_previous[16+n:, :, :, :], axis=0))/31
                    T_moving_average[n, :, :, :] = (np.sum(temperature[0:16+n, :, :, :], axis=0) + np.sum(temperature_previous[16+n:, :, :, :], axis=0))/31
                    theta_moving_average[n, :, :, :] = (np.sum(theta[0:16+n, :, :, :], axis=0) + np.sum(theta_previous[16+n:, :, :, :], axis=0))/31
                    n = n + 1
                if m < N_period and n >= 15 and n <= (N_year - 16):
                    u_moving_average[n, :, :, :] = np.mean(zonal_wind[n-15:n+16, :, :, :], axis=0)
                    v_moving_average[n, :, :, :] = np.mean(meridional_wind[n-15:n+16, :, :, :], axis=0)
                    omega_moving_average[n, :, :, :] = np.mean(omega[n-15:n+16, :, :, :], axis=0)
                    T_moving_average[n, :, :, :] = np.mean(temperature[n-15:n+16, :, :, :], axis=0)
                    theta_moving_average[n, :, :, :] = np.mean(theta[n-15:n+16, :, :, :], axis=0)
                    n = n + 1
                if m < (N_period - 1) and n > (N_year - 16):
                    u_moving_average[n, :, :, :] = (np.sum(zonal_wind[n-15:, :, :, :], axis=0) + np.sum(zonal_wind_next[0:n-(N_year-16), :, :, :], axis=0))/31
                    v_moving_average[n, :, :, :] = (np.sum(meridional_wind[n-15:, :, :, :], axis=0) + np.sum(meridional_wind_next[0:n-(N_year-16), :, :, :], axis=0))/31
                    omega_moving_average[n, :, :, :] = (np.sum(omega[n-15:, :, :, :], axis=0) + np.sum(omega_next[0:n-(N_year-16), :, :, :], axis=0))/31
                    T_moving_average[n, :, :, :] = (np.sum(temperature[n-15:, :, :, :], axis=0) + np.sum(temperature_next[0:n-(N_year-16), :, :, :], axis=0))/31
                    theta_moving_average[n, :, :, :] = (np.sum(theta[n-15:, :, :, :], axis=0) + np.sum(theta_next[0:n-(N_year-16), :, :, :], axis=0))/31
                    n = n + 1
            print("monthly moving averages for horizontal wind and temperature, time: "+ctime())
                
            print(u_moving_average[0, 0, 0, 0], v_moving_average[0, 0, 0, 0], omega_moving_average[0, 0, 0, 0], T_moving_average[0, 0, 0, 0], theta_moving_average[0, 0, 0, 0])
                
            print(len(u_moving_average), len(u_moving_average[0]), len(u_moving_average[0][0]), len(u_moving_average[0][0][0]), len(v_moving_average), len(v_moving_average[0]), len(v_moving_average[0][0]), len(v_moving_average[0][0][0]))
            
            u_average = np.zeros((N_year, N_pressure, N_y), dtype=np.float32)
            v_average = np.zeros((N_year, N_pressure, N_y), dtype=np.float32)
            omega_average = np.zeros((N_year, N_pressure, N_y), dtype=np.float32)
            T_average = np.zeros((N_year, N_pressure, N_y), dtype=np.float32)
            theta_average = np.zeros((N_year, N_pressure, N_y), dtype=np.float32)

            start_index = self.N_x_global_west
            end_index = self.N_x_global_east
            print("start and end longitude index:", start_index, end_index)
                
            if start_index >= end_index:
                u_moving_average_mod = np.concatenate((u_moving_average[:, :, :, start_index:], u_moving_average[:, :, :, 0:end_index+1]), axis=3)
                v_moving_average_mod = np.concatenate((v_moving_average[:, :, :, start_index:], v_moving_average[:, :, :, 0:end_index+1]), axis=3)
                omega_moving_average_mod = np.concatenate((omega_moving_average[:, :, :, start_index:], omega_moving_average[:, :, :, 0:end_index+1]), axis=3)
                T_moving_average_mod = np.concatenate((T_moving_average[:, :, :, start_index:], T_moving_average[:, :, :, 0:end_index+1]), axis=3)
                theta_moving_average_mod = np.concatenate((theta_moving_average[:, :, :, start_index:], theta_moving_average[:, :, :, 0:end_index+1]), axis=3)
                start_index_mod = 0
                end_index_mod = N_x - start_index + end_index
                u_average[:, :, :] = np.mean(u_moving_average_mod[:, :, :, start_index_mod:end_index_mod+1], axis=3)
                v_average[:, :, :] = np.mean(v_moving_average_mod[:, :, :, start_index_mod:end_index_mod+1], axis=3)
                omega_average[:, :, :] = np.mean(omega_moving_average_mod[:, :, :, start_index_mod:end_index_mod+1], axis=3)
                T_average[:, :, :] = np.mean(T_moving_average_mod[:, :, :, start_index_mod:end_index_mod+1], axis=3)
                theta_average[:, :, :] = np.mean(theta_moving_average_mod[:, :, :, start_index_mod:end_index_mod+1], axis=3)
                print("modified start and end longitude index:", start_index_mod, end_index_mod)
                del u_moving_average, v_moving_average, omega_moving_average, T_moving_average, theta_moving_average, u_moving_average_mod, v_moving_average_mod, omega_moving_average_mod, T_moving_average_mod, theta_moving_average_mod
            else:
                u_average[:, :, :] = np.mean(u_moving_average[:, :, :, start_index:end_index+1], axis=3)
                v_average[:, :, :] = np.mean(v_moving_average[:, :, :, start_index:end_index+1], axis=3)
                omega_average[:, :, :] = np.mean(omega_moving_average[:, :, :, start_index:end_index+1], axis=3)
                T_average[:, :, :] = np.mean(T_moving_average[:, :, :, start_index:end_index+1], axis=3)
                theta_average[:, :, :] = np.mean(theta_moving_average[:, :, :, start_index:end_index+1], axis=3)
                del u_moving_average, v_moving_average, omega_moving_average, T_moving_average, theta_moving_average
            print("zonally averaged wind and temperature, time: "+ctime())
            
            u_star = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            v_star = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            omega_star = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            T_star = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            theta_star = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            f_v_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
            meridion_momentum_flux = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            meridion_heat_flux = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            omega_flux = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            theta_flux = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            
            j = 0
            while j < N_x:
                u_star[:, :, :, j] = np.subtract(zonal_wind[:, :, :, j], u_average[:, :, :])
                v_star[:, :, :, j] = np.subtract(meridional_wind[:, :, :, j], v_average[:, :, :])
                omega_star[:, :, :, j] = np.subtract(omega[:, :, :, j], omega_average[:, :, :])
                T_star[:, :, :, j] = np.subtract(temperature[:, :, :, j], T_average[:, :, :])
                theta_star[:, :, :, j] = np.subtract(theta[:, :, :, j], theta_average[:, :, :])
                j = j + 1
            meridion_momentum_flux[:, :, :, :] = np.multiply(u_star[:, :, :, :], v_star[:, :, :, :])
            meridion_heat_flux[:, :, :, :] = np.multiply(v_star[:, :, :, :], T_star[:, :, :, :])
            omega_flux[:, :, :, :] = np.multiply(u_star[:, :, :, :], omega_star[:, :, :, :])
            theta_flux[:, :, :, :] = np.multiply(omega_star[:, :, :, :], theta_star[:, :, :, :])

            if m < 2:
                meridion_momentum_flux_dec[m, :, :, :, :] = meridion_momentum_flux[N_nov:, :, :, :]
                meridion_heat_flux_dec[m, :, :, :, :] = meridion_heat_flux[N_nov:, :, :, :]
                omega_flux_dec[m, :, :, :, :] = omega_flux[N_nov:, :, :, :]
                theta_flux_dec[m, :, :, :, :] = theta_flux[N_nov:, :, :, :]
            else:
                meridion_momentum_flux_dec[0, :, :, :, :] = meridion_momentum_flux_dec[1, :, :, :, :]
                meridion_momentum_flux_dec[1, :, :, :, :] = meridion_momentum_flux[N_nov:, :, :, :]
                meridion_heat_flux_dec[0, :, :, :, :] = meridion_heat_flux_dec[1, :, :, :, :]
                meridion_heat_flux_dec[1, :, :, :, :] = meridion_heat_flux[N_nov:, :, :, :]
                omega_flux_dec[0, :, :, :, :] = omega_flux_dec[1, :, :, :, :]
                omega_flux_dec[1, :, :, :, :] = omega_flux[N_nov:, :, :, :]
                theta_flux_dec[0, :, :, :, :] = theta_flux_dec[1, :, :, :, :]
                theta_flux_dec[1, :, :, :, :] = theta_flux[N_nov:, :, :, :]
            
            if self.period == "year":
                start_time_index = 0
                end_time_index = N_year
            elif self.period == "january":
                start_time_index = 0
                end_time_index = N_jan
            elif self.period == "february":
                start_time_index = N_jan
                end_time_index = N_feb
            elif self.period == "march":
                start_time_index = N_feb
                end_time_index = N_mar
            elif self.period == "april":
                start_time_index = N_mar
                end_time_index = N_apr
            elif self.period == "may":
                start_time_index = N_apr
                end_time_index = N_may
            elif self.period == "june":
                start_time_index = N_may
                end_time_index = N_jun
            elif self.period == "july":
                start_time_index = N_jun
                end_time_index = N_jul
            elif self.period == "august":
                start_time_index = N_jul
                end_time_index = N_aug
            elif self.period == "september":
                start_time_index = N_aug
                end_time_index = N_sep
            elif self.period == "october":
                start_time_index = N_sep
                end_time_index = N_oct
            elif self.period == "november":
                start_time_index = N_oct
                end_time_index = N_nov
            elif self.period == "december":
                start_time_index = N_nov
                end_time_index = N_year
            elif self.period == "winter":
                start_time_index = N_nov
                end_time_index = N_feb
            elif self.period == "spring":
                start_time_index = N_feb
                end_time_index = N_may
            elif self.period == "summer":
                start_time_index = N_may
                end_time_index = N_aug
            elif self.period == "autumn":
                start_time_index = N_aug
                end_time_index = N_nov
            print("start and end time index:", start_time_index, end_time_index)

            if self.period == "winter":
                if m > 0:
                    zonal_u_winter = np.concatenate((zonal_wind_previous[:, :, :, :], zonal_wind[0:N_feb, :, :, :]), axis=0)
                    zonal_u_time_zonal_average = zonal_u_winter[:, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                    meridional_v_winter = np.concatenate((meridional_wind_previous[:, :, :, :], meridional_wind[0:N_feb, :, :, :]), axis=0)
                    meridional_v_time_zonal_average = meridional_v_winter[:, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                    omega_winter = np.concatenate((omega_previous[:, :, :, :], omega[0:N_feb, :, :, :]), axis=0)
                    omega_time_zonal_average = omega_winter[:, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                    temperature_winter = np.concatenate((temperature_previous[:, :, :, :], temperature[0:N_feb, :, :, :]), axis=0)
                    temperature_time_zonal_average = temperature_winter[:, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                    f_v_time_zonal_average[:, :] = (Cor_param*np.sin(radian_constant*lat[0:N_y])*meridional_v_time_zonal_average[:, :])
                    meridion_momentum_flux_winter = np.concatenate((meridion_momentum_flux_dec[0, :, :, :, :], meridion_momentum_flux[0:N_feb, :, :, :]), axis=0)
                    meridion_momentum_flux_time_zonal_average = meridion_momentum_flux_winter[:, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                    meridion_heat_flux_winter = np.concatenate((meridion_heat_flux_dec[0, :, :, :, :], meridion_heat_flux[0:N_feb, :, :, :]), axis=0)
                    meridion_heat_flux_time_zonal_average = meridion_heat_flux_winter[:, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                    omega_flux_winter = np.concatenate((omega_flux_dec[0, :, :, :, :], omega_flux[0:N_feb, :, :, :]), axis=0)
                    omega_flux_time_zonal_average = omega_flux_winter[:, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                    theta_flux_winter = np.concatenate((theta_flux_dec[0, :, :, :, :], theta_flux[0:N_feb, :, :, :]), axis=0)
                    theta_flux_time_zonal_average = theta_flux_winter[:, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                else:
                    zonal_u_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
                    meridional_v_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
                    omega_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
                    temperature_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
                    f_v_time_zonal_average[:, :] = (Cor_param*np.sin(radian_constant*lat[0:N_y])*meridional_v_time_zonal_average[:, :])
                    meridion_momentum_flux_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
                    meridion_heat_flux_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
                    omega_flux_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
                    theta_flux_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
            else:
                zonal_u_time_zonal_average = zonal_wind[start_time_index:end_time_index, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                meridional_v_time_zonal_average = meridional_wind[start_time_index:end_time_index, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                omega_time_zonal_average = omega[start_time_index:end_time_index, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                temperature_time_zonal_average = temperature[start_time_index:end_time_index, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                f_v_time_zonal_average[:, :] = (Cor_param*np.sin(radian_constant*lat[0:N_y])*meridional_v_time_zonal_average[:, :])
                meridion_momentum_flux_time_zonal_average = meridion_momentum_flux[start_time_index:end_time_index, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                meridion_heat_flux_time_zonal_average = meridion_heat_flux[start_time_index:end_time_index, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                omega_flux_time_zonal_average = omega_flux[start_time_index:end_time_index, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
                theta_flux_time_zonal_average = theta_flux[start_time_index:end_time_index, :, :, start_index:end_index+1].mean(axis=3).mean(axis=0)
            print("start and end longitude index:", start_index, end_index)
                
            print("computation of quantities necessary for friction, time: "+ctime())

            if self.period == "winter":
                if m > 0:
                    meridion_heat_flux_averaged_yearly[m, :, :] = (meridion_heat_flux_time_zonal_average[:, :])/(10**(1))
                    meridion_momentum_flux_averaged_yearly[m, :, :] = (meridion_momentum_flux_time_zonal_average[:, :])/(10**(1))
                    omega_flux_averaged_yearly[m, :, :] = (omega_flux_time_zonal_average[:, :])/(10**(-1))
                    theta_flux_averaged_yearly[m, :, :] = (theta_flux_time_zonal_average[:, :])/(10**(-1))
            else:
                meridion_heat_flux_averaged_yearly[m, :, :] = (meridion_heat_flux_time_zonal_average[:, :])/(10**(1))
                meridion_momentum_flux_averaged_yearly[m, :, :] = (meridion_momentum_flux_time_zonal_average[:, :])/(10**(1))
                omega_flux_averaged_yearly[m, :, :] = (omega_flux_time_zonal_average[:, :])/(10**(-1))
                theta_flux_averaged_yearly[m, :, :] = (theta_flux_time_zonal_average[:, :])/(10**(-1))

            del u_average, v_average, omega_average, T_average, theta_average, u_star, v_star, omega_star, T_star, theta_star
            del theta, meridion_momentum_flux, meridion_heat_flux, omega_flux, theta_flux
            if self.period == "winter" and m > 0:
                del zonal_u_winter, meridional_v_winter, omega_winter, temperature_winter, meridion_momentum_flux_winter, meridion_heat_flux_winter, omega_flux_winter, theta_flux_winter

            if m == (N_period - 1):
                del meridion_momentum_flux_dec, meridion_heat_flux_dec, omega_flux_dec, theta_flux_dec
#            del meridional_v_time_zonal_average

            # computation of flux term and friction
            flux_term_uv = np.zeros((N_pressure, N_y), dtype=np.float32)
            term_uv = np.zeros((N_pressure, N_y), dtype=np.float32)
            term_uw = np.zeros((N_pressure, N_y), dtype=np.float32)
            flux_term_uw = np.zeros((N_pressure, N_y), dtype=np.float32)
            
            cos_forward = np.zeros((N_y), dtype=np.float32)
            cos_current = np.zeros((N_y), dtype=np.float32)
            cos_backward = np.zeros((N_y), dtype=np.float32)
            dlat = np.zeros((N_y), dtype=np.float32)
            dp = np.zeros((N_pressure), dtype=np.float32)
            cos_forward[0:N_y-1] = np.multiply(np.cos(radian_constant*lat[1:N_y]), np.cos(radian_constant*lat[1:N_y]))
            cos_forward[N_y-1] = np.multiply(np.cos(radian_constant*lat[N_y-1]), np.cos(radian_constant*lat[N_y-1]))
            cos_backward[0] = np.multiply(np.cos(radian_constant*lat[0]), np.cos(radian_constant*lat[0]))
            cos_backward[1:N_y] = np.multiply(np.cos(radian_constant*lat[0:N_y-1]), np.cos(radian_constant*lat[0:N_y-1]))
            dlat[0] = np.subtract(radian_constant*lat[1], radian_constant*lat[0])
            dlat[1:N_y-1] = np.subtract(radian_constant*lat[2:N_y], radian_constant*lat[0:N_y-2])
            dlat[N_y-1] = np.subtract(radian_constant*lat[N_y-1], radian_constant*lat[N_y-2])
            dp[0] = np.subtract(100*pressure[1], 100*pressure[0])
            dp[1:N_pressure-1] = np.subtract(100*pressure[2:N_pressure], 100*pressure[0:N_pressure-2])
            dp[N_pressure-1] = np.subtract(100*pressure[N_pressure-1], 100*pressure[N_pressure-2])
            
            _coslat = np.zeros((N_y), dtype=np.float32)
            _coslat[0] = np.cos(0.5*(radian_constant*lat[0] + radian_constant*lat[1]))
            _coslat[1:N_y-1] = np.cos(radian_constant*lat[1:N_y-1])
            _coslat[N_y-1] = np.cos(0.5*(radian_constant*lat[N_y-1] + radian_constant*lat[N_y-2]))
            coslat, p_star = np.meshgrid(_coslat, (100*pressure)**(1 - (self.R_d/self.c_p)))
            
            cos_current[0] = np.multiply(np.cos(0.5*(radian_constant*lat[0] + radian_constant*lat[1])), np.cos(0.5*(radian_constant*lat[0] + radian_constant*lat[1])))
            cos_current[1:N_y-1] = np.multiply(np.cos(radian_constant*lat[1:N_y-1]), np.cos(radian_constant*lat[1:N_y-1]))
            cos_current[N_y-1] = np.multiply(np.cos(0.5*(radian_constant*lat[N_y-1] + radian_constant*lat[N_y-2])), np.cos(0.5*(radian_constant*lat[N_y-1] + radian_constant*lat[N_y-2])))
            
            cos2_forward, p = np.meshgrid(cos_forward, 100*pressure)
            cos2_current, p = np.meshgrid(cos_current, 100*pressure)
            cos2_backward, p = np.meshgrid(cos_backward, 100*pressure)
            delta_lat, delta_p = np.meshgrid(dlat, dp)
            
            flux_term_uv[:, 0] = (np.subtract(cos2_forward[:, 0]*meridion_momentum_flux_time_zonal_average[:, 1], cos2_backward[:, 0]*meridion_momentum_flux_time_zonal_average[:, 0]))/delta_lat[:, 0]/cos2_current[:, 0]/self.R_Earth
            flux_term_uv[:, 1:N_y-1] = (np.subtract(cos2_forward[:, 1:N_y-1]*meridion_momentum_flux_time_zonal_average[:, 2:N_y], cos2_backward[:, 1:N_y-1]*meridion_momentum_flux_time_zonal_average[:, 0:N_y-2]))/delta_lat[:, 1:N_y-1]/cos2_current[:, 1:N_y-1]/self.R_Earth
            flux_term_uv[:, N_y-1] = (np.subtract(cos2_forward[:, N_y-1]*meridion_momentum_flux_time_zonal_average[:, N_y-1], cos2_backward[:, N_y-2]*meridion_momentum_flux_time_zonal_average[:, N_y-2]))/delta_lat[:, N_y-1]/cos2_current[:, N_y-1]/self.R_Earth
            
            term_uv[:, 0] = (np.subtract(cos2_forward[:, 0]*zonal_u_time_zonal_average[:, 1]*meridional_v_time_zonal_average[:, 1], cos2_backward[:, 0]*zonal_u_time_zonal_average[:, 0]*meridional_v_time_zonal_average[:, 0]))/delta_lat[:, 0]/cos2_current[:, 0]/self.R_Earth
            term_uv[:, 1:N_y-1] = (np.subtract(cos2_forward[:, 1:N_y-1]*zonal_u_time_zonal_average[:, 2:N_y]*meridional_v_time_zonal_average[:, 2:N_y], cos2_backward[:, 1:N_y-1]*zonal_u_time_zonal_average[:, 0:N_y-2]*meridional_v_time_zonal_average[:, 0:N_y-2]))/delta_lat[:, 1:N_y-1]/cos2_current[:, 1:N_y-1]/self.R_Earth
            term_uv[:, N_y-1] = (np.subtract(cos2_forward[:, N_y-1]*zonal_u_time_zonal_average[:, N_y-1]*meridional_v_time_zonal_average[:, N_y-1], cos2_backward[:, N_y-2]*zonal_u_time_zonal_average[:, N_y-2]*meridional_v_time_zonal_average[:, N_y-2]))/delta_lat[:, N_y-1]/cos2_current[:, N_y-1]/self.R_Earth
            
            term_uw[0, :] = (np.subtract(zonal_u_time_zonal_average[1, :]*omega_time_zonal_average[1, :], zonal_u_time_zonal_average[0, :]*omega_time_zonal_average[0, :]))/delta_p[0, :]
            term_uw[1:N_pressure-1, :] = (np.subtract(zonal_u_time_zonal_average[2:N_pressure, :]*omega_time_zonal_average[2:N_pressure, :], zonal_u_time_zonal_average[0:N_pressure-2, :]*omega_time_zonal_average[0:N_pressure-2, :]))/delta_p[1:N_pressure-1, :]
            term_uw[N_pressure-1, :] = (np.subtract(zonal_u_time_zonal_average[N_pressure-1, :]*omega_time_zonal_average[N_pressure-1, :], zonal_u_time_zonal_average[N_pressure-2, :]*omega_time_zonal_average[N_pressure-2, :]))/delta_p[N_pressure-1, :]
            
            flux_term_uw[0, :] = (np.subtract(omega_flux_time_zonal_average[1, :], omega_flux_time_zonal_average[0, :]))/delta_p[0, :]
            flux_term_uw[1:N_pressure-1, :] = (np.subtract(omega_flux_time_zonal_average[2:N_pressure, :], omega_flux_time_zonal_average[0:N_pressure-2, :]))/delta_p[1:N_pressure-1, :]
            flux_term_uw[N_pressure-1, :] = (np.subtract(omega_flux_time_zonal_average[N_pressure-1, :], omega_flux_time_zonal_average[N_pressure-2, :]))/delta_p[N_pressure-1, :]

            friction = (flux_term_uv + term_uv + term_uw + flux_term_uw - f_v_time_zonal_average)/(10**(-4))
                
            del flux_term_uv, term_uv, term_uw, flux_term_uw, f_v_time_zonal_average
                
            print("computation of flux term and friction, time: "+ctime())
                
            del cos_forward, cos_current, cos_backward, dlat, dp, delta_p, omega_time_zonal_average
                
            print(len(friction), len(friction[35]))
            
            friction_averaged_yearly[m, :, :] = friction[:, :]
            
            print(len(friction_averaged_yearly), len(friction_averaged_yearly[m]), len(friction_averaged_yearly[m][35]))

            # computation of terms of diabatic heating
            advection_term_2 = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            i = 0
            while i < N_y:
                if i == 0:
                    dT = np.subtract(temperature[:, :, i+1, :], temperature[:, :, i, :])
                    dlat = np.subtract(radian_constant*lat[i+1], radian_constant*lat[i])
                if i == (N_y - 1):
                    dT = np.subtract(temperature[:, :, i, :], temperature[:, :, i-1, :])
                    dlat = np.subtract(radian_constant*lat[i], radian_constant*lat[i-1])
                if i > 0 and i < (N_y - 1):
                    dT = np.subtract(temperature[:, :, i+1, :], temperature[:, :, i-1, :])
                    dlat = np.subtract(radian_constant*lat[i+1], radian_constant*lat[i-1])
                advection_term_2[:, :, i, :] = meridional_wind[:, :, i, :]*dT/dlat/self.R_Earth
                i = i + 1
            print("third term of diabatic heating, time: "+ctime())
                
            del dT, dlat, meridional_wind
                
            print(len(advection_term_2), len(advection_term_2[0]), len(advection_term_2[0][0]), len(advection_term_2[0][0][0]))
            
            dT_dt = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
                
            n = 0
            while n < N_year:
                if m == 0 and n == 0:
                    deltaT = np.subtract(temperature[n+1, :, :, :], temperature[n, :, :, :])
                    dT_dt[n, :, :, :] = (deltaT/(24*3600))
                if n > 0 and n < (N_year - 1):
                    deltaT = np.subtract(temperature[n+1, :, :, :], temperature[n-1, :, :, :])
                    dT_dt[n, :, :, :] = (deltaT/(2*24*3600))
                if m < (N_period - 1) and n == (N_year - 1):
                    deltaT = np.subtract(temperature_next[0, :, :, :], temperature[n-1, :, :, :])
                    dT_dt[n, :, :, :] = (deltaT/(2*24*3600))
                if m > 0 and n == 0:
                    deltaT = np.subtract(temperature[n+1, :, :, :], temperature_previous[-1, :, :, :])
                    dT_dt[n, :, :, :] = (deltaT/(2*24*3600))
                if m == (N_period - 1) and n == (N_year - 1):
                    deltaT = np.subtract(temperature[n, :, :, :], temperature[n-1, :, :, :])
                    dT_dt[n, :, :, :] = (deltaT/(24*3600))
                n = n + 1
            print("first term of diabatic heating, time: "+ctime())
                
            del deltaT
            if m > 0:
                del zonal_wind_previous, meridional_wind_previous, temperature_previous
            if m < (N_period - 1):
                del zonal_wind_next, meridional_wind_next, temperature_next
                
            print(len(dT_dt), len(dT_dt[300]), len(dT_dt[300][35]), len(dT_dt[300][35][80]))
            
            diabatic_heating = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            diabatic_heating[:, :, :, :] = np.add(dT_dt[:, :, :, :], advection_term_2[:, :, :, :])
            print("first step in computation of diabatic heating, time: "+ctime())
                
            del dT_dt, advection_term_2
            
            # computation of terms of diabatic heating
            advection_term_1 = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            j = 0
            while j < N_x:
                if j == 0:
                    dT = np.subtract(temperature[:, :, :, j+1], temperature[:, :, :, -1])
                    dlon = np.subtract(radian_constant*lon[j+1], radian_constant*lon[-1])
                if j == N_x - 1:
                    dT = np.subtract(temperature[:, :, :, 0], temperature[:, :, :, j-1])
                    dlon = np.subtract(radian_constant*lon[0], radian_constant*lon[j-1])
                if j > 0 and j < N_x - 1:
                    dT = np.subtract(temperature[:, :, :, j+1], temperature[:, :, :, j-1])
                    dlon = np.subtract(radian_constant*lon[j+1], radian_constant*lon[j-1])
                advection_term_1[:, :, :, j] = zonal_wind[:, :, :, j]*dT/dlon/coslat/self.R_Earth
                j = j + 1
            print("second term of diabatic heating, time: "+ctime())
                
            del dT, dlon, zonal_wind
                
            print(len(advection_term_1), len(advection_term_1[0]), len(advection_term_1[0][0]), len(advection_term_1[0][0][0]))
            
            diabatic_heating[:, :, :, :] = np.add(diabatic_heating[:, :, :, :], advection_term_1[:, :, :, :])
            print("second step in computation of diabatic heating, time: "+ctime())
                
            del advection_term_1
            
            # computation of terms of diabatic heating
            advection_term_3 = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            k = 0
            while k < N_pressure:
                if k == 0:
                    dT = np.subtract(temperature[:, k+1, :, :], temperature[:, k, :, :])
                    dp = np.subtract(100*pressure[k+1], 100*pressure[k])
                if k == N_pressure - 1:
                    dT = np.subtract(temperature[:, k, :, :], temperature[:, k-1, :, :])
                    dp = np.subtract(100*pressure[k], 100*pressure[k-1])
                if k > 0 and k < N_pressure - 1:
                    dT = np.subtract(temperature[:, k+1, :, :], temperature[:, k-1, :, :])
                    dp = np.subtract(100*pressure[k+1], 100*pressure[k-1])
                advection_term_3[:, k, :, :] = omega[:, k, :, :]*dT/dp
                k = k + 1
            print("fourth term of diabatic heating, time: "+ctime())
                
            del dT, dp
                
            print(len(advection_term_3), len(advection_term_3[0]), len(advection_term_3[0][0]), len(advection_term_3[0][0][0]))
            
            diabatic_heating[:, :, :, :] = np.add(diabatic_heating[:, :, :, :], advection_term_3[:, :, :, :])
            print("third step in computation of diabatic heating, time: "+ctime())
                
            del advection_term_3
                    
            # computation of terms of diabatic heating
            term_5 = np.zeros((N_year, N_pressure, N_y, N_x), dtype=np.float32)
            k = 0
            while k < N_pressure:
                term_5[:, k, :, :] = (self.R_d/self.c_p)*omega[:, k, :, :]*temperature[:, k, :, :]/(100*pressure[k])
                k = k + 1
            print("fifth term of diabatic heating, time: "+ctime())
                
            del omega, temperature
                
            print(len(term_5), len(term_5[0]), len(term_5[0][0]), len(term_5[0][0][0]))
            
            # computation of diabatic heating
            diabatic_heating_zonally = np.zeros((N_year, N_pressure, N_y), dtype=np.float32)
            diabatic_heating[:, :, :, :] = np.subtract(diabatic_heating[:, :, :, :], term_5[:, :, :, :])
            
            if start_index >= end_index:
                diabatic_heating_mod = np.concatenate((diabatic_heating[:, :, :, start_index:], diabatic_heating[:, :, :, 0:end_index+1]), axis=3)
                start_index_mod = 0
                end_index_mod = N_x - start_index + end_index
                diabatic_heating_zonally[:, :, :] = np.mean(diabatic_heating_mod[:, :, :, start_index_mod:end_index_mod+1], axis=3)
            else:
                diabatic_heating_zonally[:, :, :] = np.mean(diabatic_heating[:, :, :, start_index:end_index+1], axis=3)
                if m < 2:
                    diabatic_heating_zonally_dec[m, :, :, :] = np.mean(diabatic_heating[N_nov:, :, :, start_index:end_index+1], axis=3)
                else:
                    diabatic_heating_zonally_dec[0, :, :, :] = diabatic_heating_zonally_dec[1, :, :, :]
                    diabatic_heating_zonally_dec[1, :, :, :] = np.mean(diabatic_heating[N_nov:, :, :, start_index:end_index+1], axis=3)
            print("fourth step in computation of diabatic heating and zonal averaging, time: "+ctime())
                
            del term_5
            
            # time averaging of diabatic heating
            diabatic_heating_time_zonal_average = np.zeros((N_pressure, N_y), dtype=np.float32)
            if self.period == "winter":
                if m > 0:
                    diabatic_heating_winter = np.concatenate((diabatic_heating_zonally_dec[0, :, :, :], diabatic_heating_zonally[0:N_feb, :, :]), axis=0)
                    diabatic_heating_time_zonal_average[:, :] = ((np.mean(diabatic_heating_winter[:, :, :], axis=0))/(10**(-5)))
            else:
                diabatic_heating_time_zonal_average[:, :] = ((np.mean(diabatic_heating_zonally[start_time_index:end_time_index, :, :], axis=0))/(10**(-5)))
            print("time averaging of diabatic heating by year, time: "+ctime())
            
            del diabatic_heating, diabatic_heating_zonally
            if self.period == "winter" and m > 0:
                del diabatic_heating_winter
            if m == (N_period - 1):
                del diabatic_heating_zonally_dec
                
            print(len(diabatic_heating_time_zonal_average), len(diabatic_heating_time_zonal_average[35]))
            
            diabatic_heating_averaged_yearly[m, :, :] = diabatic_heating_time_zonal_average[:, :]
            
            print(len(diabatic_heating_averaged_yearly), len(diabatic_heating_averaged_yearly[m]), len(diabatic_heating_averaged_yearly[m][35]))

            print("end of computation for year "+str(int(start_period)+m)+", time: "+ctime())
            
            print("start of computation of streamfunction for year "+str(int(start_period)+m)+", time: "+ctime())
            
            dp1 = np.zeros((N_pressure), dtype=np.float32)
            dp2 = np.zeros((N_pressure), dtype=np.float32)
            dp3 = np.zeros((N_pressure), dtype=np.float32)
            dp4 = np.zeros((N_pressure), dtype=np.float32)
            p_plus = np.zeros((N_pressure), dtype=np.float32)
            p_minus = np.zeros((N_pressure), dtype=np.float32)
            dp1[N_pressure-1] = 0 - 100*pressure[N_pressure-1]
            dp1[0:N_pressure-1] = 100*pressure[1:N_pressure] - 100*pressure[0:N_pressure-1]
            print(dp1)
            p_plus[0:N_pressure-1] = 0.5*(100*pressure[0:N_pressure-1] + 100*pressure[1:N_pressure])
            p_plus[N_pressure-1] = 0.5*(100*pressure[N_pressure-1])
            p_minus[0] = 100*pressure[0]
            p_minus[1:N_pressure] = 0.5*(100*pressure[1:N_pressure] + 100*pressure[0:N_pressure-1])
            dp2[0:N_pressure] = p_plus[0:N_pressure] - p_minus[0:N_pressure]
            print(dp2)
            dp3[0] = 100*pressure[1] - 100*pressure[0]
            dp3[N_pressure-1] = 0 - 100*pressure[N_pressure-2]
            dp3[1:N_pressure-1] = 100*pressure[2:N_pressure] - 100*pressure[0:N_pressure-2]
            print(dp3)
            #dp4[0] = 0
            dp4[0] = 100*pressure[0] - 100*(pressure[0] + 25)
            dp4[1:N_pressure] = 100*pressure[1:N_pressure] - 100*pressure[0:N_pressure-1]
            print(dp4)
            dlat1 = np.zeros((N_y), dtype=np.float32)
            dlat2 = np.zeros((N_y), dtype=np.float32)
            dlat3 = np.zeros((N_y), dtype=np.float32)
            dlat4 = np.zeros((N_y), dtype=np.float32)
            lat_plus = np.zeros((N_y), dtype=np.float32)
            lat_minus = np.zeros((N_y), dtype=np.float32)
            dlat1[N_y-1] = 0
            dlat1[0:N_y-1] = radian_constant*lat[1:N_y] - radian_constant*lat[0:N_y-1]
            print(dlat1)
            lat_plus[0:N_y-1] = 0.5*(radian_constant*lat[0:N_y-1] + radian_constant*lat[1:N_y])
            lat_plus[N_y-1] = 0.5*(radian_constant*lat[N_y-1] + radian_constant*(lat[1] + 180))
            lat_minus[0] = 0.5*(radian_constant*lat[0] + radian_constant*(lat[N_y-2] - 180))
            lat_minus[1:N_y] = 0.5*(radian_constant*lat[1:N_y] + radian_constant*lat[0:N_y-1])
            dlat2[0:N_y] = lat_plus[0:N_y] - lat_minus[0:N_y]
            print(dlat2)
            dlat3[0] = radian_constant*lat[1] - radian_constant*lat[0]
            dlat3[N_y-1] = radian_constant*lat[N_y-1] - radian_constant*lat[N_y-2]
            dlat3[1:N_y-1] = radian_constant*lat[2:N_y] - radian_constant*lat[0:N_y-2]
            print(dlat3)
            dlat4[0] = 0
            dlat4[1:N_y] = radian_constant*lat[1:N_y] - radian_constant*lat[0:N_y-1]
            print(dlat4)
            
            lat_1, p_1 = np.meshgrid(dlat1, dp1)
            lat_2, p_2 = np.meshgrid(dlat2, dp2)
            lat_3, p_3 = np.meshgrid(dlat3, dp3)
            lat_4, p_4 = np.meshgrid(dlat4, dp4)
            
            dp = np.zeros((N_pressure), dtype=np.float32)
            dp[0] = 100*pressure[1] - 100*pressure[0]
            dp[1:N_pressure-1] = 100*pressure[2:N_pressure] - 100*pressure[0:N_pressure-2]
            dp[N_pressure-1] = 100*pressure[N_pressure-1] - 100*pressure[N_pressure-2]
            
            f_C = Cor_param*np.sin(radian_constant*lat[:])
            f_Cor, delta_p = np.meshgrid(f_C, dp)
            
            cos_lat, p = np.meshgrid(np.cos(radian_constant*lat[:]), 100*pressure)
            
            _coslat = np.zeros((N_y), dtype=np.float32)
            _coslat[0] = np.cos(0.5*(radian_constant*lat[0] + radian_constant*lat[1]))
            _coslat[1:N_y-1] = np.cos(radian_constant*lat[1:N_y-1])
            _coslat[N_y-1] = np.cos(0.5*(radian_constant*lat[N_y-1] + radian_constant*lat[N_y-2]))
            coslat, p_star = np.meshgrid(_coslat, (100*pressure)**(1 - (self.R_d/self.c_p)))
            
            dlat = np.zeros((N_y), dtype=np.float32)
            dlat[0:N_y] = radian_constant
            print(dlat)
            delta_lat, p_star = np.meshgrid(dlat, (100*pressure)**(1 - (self.R_d/self.c_p)))
            
            coslat_plus = np.zeros((N_pressure, N_y), dtype=np.float32)
            coslat_minus = np.zeros((N_pressure, N_y), dtype=np.float32)
            coslat_plus[:, 0:N_y-1] = 0.5*(cos_lat[:, 1:N_y] + cos_lat[:, 0:N_y-1])
            coslat_plus[:, N_y-1] = 0.5*(cos_lat[:, N_y-1] - cos_lat[:, N_y-2])
            coslat_minus[:, 0] = 0.5*(cos_lat[:, 0] - cos_lat[:, 1])
            coslat_minus[:, 1:N_y] = 0.5*(cos_lat[:, 1:N_y] + cos_lat[:, 0:N_y-1])
            
            print(f_Cor)
            print(cos_lat)
            print(p)
            print(p_star)

            del dp1, dp2, dp3, dp4, p_plus, p_minus, dlat1, dlat2, dlat3, dlat4, lat_plus, lat_minus, dp, f_C, dlat
            
            factor_a = f_Cor*f_Cor*self.g/2./np.pi/self.R_Earth/coslat/p_1/p_2
            print("factor a, time: "+ctime())
            
            print(len(factor_a), len(factor_a[0]))

            dT = np.zeros((N_pressure, N_y), dtype=np.float32)
            dT[0, :] = np.subtract(temperature_time_zonal_average[1, :], temperature_time_zonal_average[0, :])
            dT[1:N_pressure-1, :] = np.subtract(temperature_time_zonal_average[2:N_pressure, :], temperature_time_zonal_average[0:N_pressure-2, :])
            dT[N_pressure-1, :] = np.subtract(temperature_time_zonal_average[N_pressure-1, :], temperature_time_zonal_average[N_pressure-2, :])
            
            term_T_vs_p = np.subtract((dT/delta_p), ((self.R_d/self.c_p)*temperature_time_zonal_average/p))
            
            deltaT = np.zeros((N_pressure, N_y), dtype=np.float32)
            deltaT[:, 0] = np.subtract((term_T_vs_p[:, 1]/coslat[:, 1]), (term_T_vs_p[:, 0]/coslat[:, 0]))
            deltaT[:, 1:N_y-1] = np.subtract((term_T_vs_p[:, 2:N_y]/coslat[:, 2:N_y]), (term_T_vs_p[:, 0:N_y-2]/coslat[:, 0:N_y-2]))
            deltaT[:, N_y-1] = np.subtract((term_T_vs_p[:, N_y-1]/coslat[:, N_y-1]), (term_T_vs_p[:, N_y-2]/coslat[:, N_y-2]))
            
            factor_b = -1.*self.g*self.R_d*deltaT/2./np.pi/self.R_Earth/self.R_Earth/self.R_Earth/p/lat_3/lat_3
            print("factor b, time: "+ctime())
            
            del dT, deltaT
            
            print(len(factor_b), len(factor_b[0])),
            
            factor_c = -1.*self.g*self.R_d*term_T_vs_p/2./np.pi/self.R_Earth/self.R_Earth/self.R_Earth/p/coslat/delta_lat/delta_lat
            print("factor c, time: "+ctime())
            
            print(len(factor_c), len(factor_c[0]))
            
            del term_T_vs_p
            
            factor_d = f_Cor*f_Cor*self.g/2./np.pi/self.R_Earth/coslat/p_2/p_4
            print("factor d, time: "+ctime())
            
            print(len(factor_d), len(factor_d[0]))
            
            dT = np.zeros((N_pressure, N_y), dtype=np.float32)
            dT[:, 0] = np.subtract(temperature_time_zonal_average[:, 1], temperature_time_zonal_average[:, 0])
            dT[:, 1:N_y-1] = np.subtract(temperature_time_zonal_average[:, 2:N_y], temperature_time_zonal_average[:, 0:N_y-2])
            dT[:, N_y-1] = np.subtract(temperature_time_zonal_average[:, N_y-1], temperature_time_zonal_average[:, N_y-2])

            factor_A1 = self.g*self.R_d*dT/2./np.pi/self.R_Earth/self.R_Earth/self.R_Earth/p/coslat/lat_3/lat_3/p_3
            print("factor A1, time: "+ctime())

            print(len(factor_A1), len(factor_A1[0]))
            
            dcoslat = np.zeros((N_pressure, N_y), dtype=np.float32)
            dcoslat[:, 0] = np.subtract(1./coslat[:, 1], 1./coslat[:, 0])
            dcoslat[:, 1:N_y-1] = np.subtract(1./coslat[:, 2:N_y], 1./coslat[:, 0:N_y-2])
            dcoslat[:, N_y-1] = np.subtract(1./coslat[:, N_y-1], 1./coslat[:, N_y-2])
            
            factor_A2 = self.g*self.R_d*dT*dcoslat/2./np.pi/self.R_Earth/self.R_Earth/self.R_Earth/p/lat_3/lat_3/p_3
            print("factor A2, time: "+ctime())
            
            print(len(factor_A2), len(factor_A2[0]))
            
            del dT, dcoslat

            dT1 = np.zeros((N_pressure, N_y), dtype=np.float32)
            dT2 = np.zeros((N_pressure, N_y), dtype=np.float32)
            dT1[:, 0:N_y-1] = np.subtract(temperature_time_zonal_average[:, 1:N_y], temperature_time_zonal_average[:, 0:N_y-1])
            dT1[:, N_y-1] = 0
            dT2[:, 0] = 0
            dT2[:, 1:N_y] = np.subtract(temperature_time_zonal_average[:, 1:N_y], temperature_time_zonal_average[:, 0:N_y-1])

            factor_A3 = self.g*self.R_d*(dT1 - dT2)/2./np.pi/self.R_Earth/self.R_Earth/self.R_Earth/p/coslat/delta_lat/delta_lat/p_3
            print("factor A3, time: "+ctime())

            print(len(factor_A3), len(factor_A3[0]))

            del dT1, dT2, temperature_time_zonal_average

            du = np.zeros((N_pressure, N_y), dtype=np.float32)
            du[0, :] = np.subtract(zonal_u_time_zonal_average[1, :], zonal_u_time_zonal_average[0, :])
            du[1:N_pressure-1, :] = np.subtract(zonal_u_time_zonal_average[2:N_pressure, :], zonal_u_time_zonal_average[0:N_pressure-2, :])
            du[N_pressure-1, :] = np.subtract(zonal_u_time_zonal_average[N_pressure-1, :], zonal_u_time_zonal_average[N_pressure-2, :])

            factor_B1 = f_Cor*self.g*du/2./np.pi/self.R_Earth/self.R_Earth/coslat/lat_3/p_3/p_3
            print("factor B1, time: "+ctime())

            del du

            print(len(factor_B1), len(factor_B1[0]))

            du = np.zeros((N_pressure, N_y), dtype=np.float32)
            du[0:N_pressure-1, :] = np.subtract(zonal_u_time_zonal_average[1:N_pressure, :], zonal_u_time_zonal_average[0:N_pressure-1, :])
            du[N_pressure-1, :] = 0

            factor_B2 = f_Cor*self.g*du/2./np.pi/self.R_Earth/self.R_Earth/coslat/p_1/p_2/lat_3
            print("factor B2, time: "+ctime())

            del du

            print(len(factor_B2), len(factor_B2[0]))

            du = np.zeros((N_pressure, N_y), dtype=np.float32)
            du[0, :] = 0
            du[1:N_pressure, :] = np.subtract(zonal_u_time_zonal_average[1:N_pressure, :], zonal_u_time_zonal_average[0:N_pressure-1, :])

            factor_B3 = f_Cor*self.g*du/2./np.pi/self.R_Earth/self.R_Earth/coslat/p_2/p_4/lat_3
            print("factor B5, time: "+ctime())

            del du

            print(len(factor_B3), len(factor_B3[0]))
            
            dflux1 = np.zeros((N_pressure, N_y), dtype=np.float32)
            dflux2 = np.zeros((N_pressure, N_y), dtype=np.float32)

            dflux1[0, :] = 0
            dflux1[:, 0] = 0
            dflux1[1:N_pressure-1, 1:N_y-1] = np.subtract(cos_lat[2:N_pressure, 2:N_y]*zonal_u_time_zonal_average[2:N_pressure, 2:N_y], cos_lat[0:N_pressure-2, 2:N_y]*zonal_u_time_zonal_average[0:N_pressure-2, 2:N_y])
            dflux1[N_pressure-1, :] = 0
            dflux1[:, N_y-1] = 0

            dflux2[0, :] = 0
            dflux2[:, 0] = 0
            dflux2[1:N_pressure-1, 1:N_y-1] = np.subtract(cos_lat[2:N_pressure, 0:N_y-2]*zonal_u_time_zonal_average[2:N_pressure, 0:N_y-2], cos_lat[0:N_pressure-2, 0:N_y-2]*zonal_u_time_zonal_average[0:N_pressure-2, 0:N_y-2])
            dflux2[N_pressure-1, :] = 0
            dflux2[:, N_y-1] = 0

            factor_C1 = -1.*f_Cor*self.g*(dflux1 - dflux2)/2./np.pi/self.R_Earth/self.R_Earth/cos2_current/lat_3/p_3/p_3
            print("factor C1, time: "+ctime())

            del dflux1, dflux2

            print(len(factor_C1), len(factor_C1[0]))

            dflux = np.zeros((N_pressure, N_y), dtype=np.float32)
            dflux[:, 0] = np.subtract((zonal_u_time_zonal_average[:, 1]*cos_lat[:, 1]), (zonal_u_time_zonal_average[:, 0]*cos_lat[:, 0]))
            dflux[:, 1:N_y-1] = np.subtract((zonal_u_time_zonal_average[:, 2:N_y]*cos_lat[:, 2:N_y]), (zonal_u_time_zonal_average[:, 0:N_y-2]*cos_lat[:, 0:N_y-2]))
            dflux[:, N_y-1] = np.subtract((zonal_u_time_zonal_average[:, N_y-1]*cos_lat[:, N_y-1]), (zonal_u_time_zonal_average[:, N_y-2]*cos_lat[:, N_y-2]))

            factor_C2 = -1.*f_Cor*self.g*dflux/2./np.pi/self.R_Earth/self.R_Earth/cos2_current/p_1/p_2/lat_3
            print("factor C2, time: "+ctime())

            print(len(factor_C2), len(factor_C2[0]))

            factor_C3 = -1.*f_Cor*self.g*dflux/2./np.pi/self.R_Earth/self.R_Earth/cos2_current/p_2/p_4/lat_3
            print("factor C3, time: "+ctime())

            del dflux, zonal_u_time_zonal_average

            print(len(factor_C3), len(factor_C3[0]))
            
            linear_operator_L = np.zeros((dim, dim), dtype=np.float32)
            linear_operator_L_S2 = np.zeros((dim, dim), dtype=np.float32)
            linear_operator_L_u = np.zeros((dim, dim), dtype=np.float32)
            linear_operator_L_T = np.zeros((dim, dim), dtype=np.float32)
            
            d = 0
            while d < dim:
                linear_operator_L[d, d] = 10**(-32)
                linear_operator_L_S2[d, d] = 10**(-32)
                linear_operator_L_u[d, d] = 10**(-32)
                linear_operator_L_T[d, d] = 10**(-32)
                d = d + 1
            print("linear differential operator L, time: "+ctime())
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    if k > 0 and k < (N_pressure - 1) and i > 0 and i < (N_y - 1):
                        linear_operator_L[i+k*N_y, (i-1)+(k-1)*N_y] = linear_operator_L[i+k*N_y, (i-1)+(k-1)*N_y] + (factor_A1[k, i] + factor_B1[k, i])
                        linear_operator_L[i+k*N_y, i+(k-1)*N_y] = linear_operator_L[i+k*N_y, i+(k-1)*N_y] + (factor_d[k, i] - factor_A2[k, i] - factor_A3[k, i] - factor_C1[k, i] + factor_C3[k, i])
                        linear_operator_L[i+k*N_y, (i+1)+(k-1)*N_y] = linear_operator_L[i+k*N_y, (i+1)+(k-1)*N_y] + (0 - factor_A1[k, i] - factor_B1[k, i])
                        linear_operator_L[i+k*N_y, (i-1)+k*N_y] = linear_operator_L[i+k*N_y, (i-1)+k*N_y] + (factor_c[k, i] - factor_b[k, i] - factor_B2[k, i] + factor_B3[k, i])
                        linear_operator_L[i+k*N_y, i+k*N_y] = linear_operator_L[i+k*N_y, i+k*N_y] - (factor_a[k, i] + 2.*factor_c[k, i] + factor_d[k, i] + factor_C2[k, i] + factor_C3[k, i])
                        linear_operator_L[i+k*N_y, (i+1)+k*N_y] = linear_operator_L[i+k*N_y, (i+1)+k*N_y] + (factor_b[k, i] + factor_c[k, i] + factor_B2[k, i] - factor_B3[k, i])
                        linear_operator_L[i+k*N_y, (i-1)+(k+1)*N_y] = linear_operator_L[i+k*N_y, (i-1)+(k+1)*N_y] + (0 - factor_A1[k, i] - factor_B1[k, i])
                        linear_operator_L[i+k*N_y, i+(k+1)*N_y] = linear_operator_L[i+k*N_y, i+(k+1)*N_y] + (factor_a[k, i] + factor_A2[k, i] + factor_A3[k, i] + factor_C1[k, i] + factor_C2[k, i])
                        linear_operator_L[i+k*N_y, (i+1)+(k+1)*N_y] = linear_operator_L[i+k*N_y, (i+1)+(k+1)*N_y] + (factor_A1[k, i] + factor_B1[k, i])
                        #
                        linear_operator_L_S2[i+k*N_y, (i-1)+(k-1)*N_y] = linear_operator_L_S2[i+k*N_y, (i-1)+(k-1)*N_y]
                        linear_operator_L_S2[i+k*N_y, i+(k-1)*N_y] = linear_operator_L_S2[i+k*N_y, i+(k-1)*N_y]
                        linear_operator_L_S2[i+k*N_y, (i+1)+(k-1)*N_y] = linear_operator_L_S2[i+k*N_y, (i+1)+(k-1)*N_y]
                        linear_operator_L_S2[i+k*N_y, (i-1)+k*N_y] = linear_operator_L_S2[i+k*N_y, (i-1)+k*N_y] + (factor_c[k, i] - factor_b[k, i])
                        linear_operator_L_S2[i+k*N_y, i+k*N_y] = linear_operator_L_S2[i+k*N_y, i+k*N_y] - (2.*factor_c[k, i])
                        linear_operator_L_S2[i+k*N_y, (i+1)+k*N_y] = linear_operator_L_S2[i+k*N_y, (i+1)+k*N_y] + (factor_b[k, i] + factor_c[k, i])
                        linear_operator_L_S2[i+k*N_y, (i-1)+(k+1)*N_y] = linear_operator_L_S2[i+k*N_y, (i-1)+(k+1)*N_y]
                        linear_operator_L_S2[i+k*N_y, i+(k+1)*N_y] = linear_operator_L_S2[i+k*N_y, i+(k+1)*N_y]
                        linear_operator_L_S2[i+k*N_y, (i+1)+(k+1)*N_y] = linear_operator_L_S2[i+k*N_y, (i+1)+(k+1)*N_y]
                        #
                        linear_operator_L_u[i+k*N_y, (i-1)+(k-1)*N_y] = linear_operator_L_u[i+k*N_y, (i-1)+(k-1)*N_y] + factor_B1[k, i]
                        linear_operator_L_u[i+k*N_y, i+(k-1)*N_y] = linear_operator_L_u[i+k*N_y, i+(k-1)*N_y] + (0 - factor_C1[k, i] + factor_C3[k, i])
                        linear_operator_L_u[i+k*N_y, (i+1)+(k-1)*N_y] = linear_operator_L_u[i+k*N_y, (i+1)+(k-1)*N_y] - factor_B1[k, i]
                        linear_operator_L_u[i+k*N_y, (i-1)+k*N_y] = linear_operator_L_u[i+k*N_y, (i-1)+k*N_y] + (factor_B3[k, i] - factor_B2[k, i])
                        linear_operator_L_u[i+k*N_y, i+k*N_y] = linear_operator_L_u[i+k*N_y, i+k*N_y] - (factor_C2[k, i] + factor_C3[k, i])
                        linear_operator_L_u[i+k*N_y, (i+1)+k*N_y] = linear_operator_L_u[i+k*N_y, (i+1)+k*N_y] + (factor_B2[k, i] - factor_B3[k, i])
                        linear_operator_L_u[i+k*N_y, (i-1)+(k+1)*N_y] = linear_operator_L_u[i+k*N_y, (i-1)+(k+1)*N_y] - factor_B1[k, i]
                        linear_operator_L_u[i+k*N_y, i+(k+1)*N_y] = linear_operator_L_u[i+k*N_y, i+(k+1)*N_y] + (factor_C1[k, i] + factor_C2[k, i])
                        linear_operator_L_u[i+k*N_y, (i+1)+(k+1)*N_y] = linear_operator_L_u[i+k*N_y, (i+1)+(k+1)*N_y] + factor_B1[k, i]
                        #
                        linear_operator_L_T[i+k*N_y, (i-1)+(k-1)*N_y] = linear_operator_L_T[i+k*N_y, (i-1)+(k-1)*N_y] + factor_A1[k, i]
                        linear_operator_L_T[i+k*N_y, i+(k-1)*N_y] = linear_operator_L_T[i+k*N_y, i+(k-1)*N_y] + (0 - factor_A2[k, i] - factor_A3[k, i])
                        linear_operator_L_T[i+k*N_y, (i+1)+(k-1)*N_y] = linear_operator_L_T[i+k*N_y, (i+1)+(k-1)*N_y] - factor_A1[k, i]
                        linear_operator_L_T[i+k*N_y, (i-1)+k*N_y] = linear_operator_L_T[i+k*N_y, (i-1)+k*N_y]
                        linear_operator_L_T[i+k*N_y, i+k*N_y] = linear_operator_L_T[i+k*N_y, i+k*N_y]
                        linear_operator_L_T[i+k*N_y, (i+1)+k*N_y] = linear_operator_L_T[i+k*N_y, (i+1)+k*N_y]
                        linear_operator_L_T[i+k*N_y, (i-1)+(k+1)*N_y] = linear_operator_L_T[i+k*N_y, (i-1)+(k+1)*N_y] - factor_A1[k, i]
                        linear_operator_L_T[i+k*N_y, i+(k+1)*N_y] = linear_operator_L_T[i+k*N_y, i+(k+1)*N_y] + (factor_A2[k, i] + factor_A3[k, i])
                        linear_operator_L_T[i+k*N_y, (i+1)+(k+1)*N_y] = linear_operator_L_T[i+k*N_y, (i+1)+(k+1)*N_y] + factor_A1[k, i]
                    i = i + 1
                k = k + 1
            print("construction of linear differential operator L, time: "+ctime())
            
            print(len(linear_operator_L), len(linear_operator_L[0]))
            
            print(factor_a[14, 70], factor_b[14, 70], factor_c[14, 70], factor_d[14, 70])
            print(factor_A1[14, 70], factor_A2[14, 70], factor_A3[14, 70], factor_B1[14, 70], factor_B2[14, 70], factor_B3[14, 70], factor_C1[14, 70], factor_C2[14, 70], factor_C3[14, 70])
            print(linear_operator_L[2604, 2422], linear_operator_L[2604, 2423], linear_operator_L[2604, 2424], linear_operator_L[2604, 2603], linear_operator_L[2604, 2604], linear_operator_L[2604, 2605], linear_operator_L[2604, 2784], linear_operator_L[2604, 2785], linear_operator_L[2604, 2786])
            print((factor_d[14, 70] - factor_A2[14, 70] - factor_A3[14, 70] - factor_C1[14, 70] + factor_C3[14, 70]))
            print((factor_A1[14, 70] + factor_B1[14, 70]))
            print(-1.*(factor_a[14, 70] + 2.*factor_c[14, 70] + factor_d[14, 70] + factor_C2[14, 70] + factor_C3[14, 70]))
            
            dJ = np.zeros((N_pressure, N_y), dtype=np.float32)
            dJ[:, 0] = np.subtract(10**(-5)*diabatic_heating_time_zonal_average[:, 1], 10**(-5)*diabatic_heating_time_zonal_average[:, 0])
            dJ[:, 1:N_y-1] = np.subtract(10**(-5)*diabatic_heating_time_zonal_average[:, 2:N_y], 10**(-5)*diabatic_heating_time_zonal_average[:, 0:N_y-2])
            dJ[:, N_y-1] = np.subtract(10**(-5)*diabatic_heating_time_zonal_average[:, N_y-1], 10**(-5)*diabatic_heating_time_zonal_average[:, N_y-2])

            term_1_vector_D = self.R_d*dJ/self.R_Earth/p/lat_3
            print("first term of vector D, time: "+ctime())
            
            del dJ, diabatic_heating_time_zonal_average
            
            print(len(term_1_vector_D), len(term_1_vector_D[0]))
            
            diabatic_heating_yearly[m, :, :] = term_1_vector_D

            print(len(diabatic_heating_yearly), len(diabatic_heating_yearly[m]), len(diabatic_heating_yearly[m][0]))
            
            vector_D_partial = np.zeros((dim), dtype=np.float32)
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    if k > 0 and k < (N_pressure - 1) and i > 0 and i < (N_y - 1):
                        vector_D_partial[i+k*N_y] = term_1_vector_D[k, i]
                    i = i + 1
                k = k + 1
            print("vector D1, time: "+ctime())
            
            print(term_1_vector_D[14, 70], vector_D_partial[2604])
            
            exact_solution_partial = np.dot(np.linalg.inv(linear_operator_L), vector_D_partial)
            print("exact solution for J, time: "+ctime())
            print(exact_solution_partial)
            print(len(exact_solution_partial))
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    psi_J[m, k, i] = exact_solution_partial[i+k*N_y]
                    i = i + 1
                k = k + 1
            print("exact solution for streamfunction for J for year "+str(int(start_period)+m)+", time: "+ctime())
            
            dflux1 = np.zeros((N_pressure, N_y), dtype=np.float32)
            dflux2 = np.zeros((N_pressure, N_y), dtype=np.float32)
            dflux1[:, 0:N_y-1] = np.subtract(cos_lat[:, 1:N_y]*meridion_heat_flux_time_zonal_average[:, 1:N_y], cos_lat[:, 0:N_y-1]*meridion_heat_flux_time_zonal_average[:, 0:N_y-1])
            dflux1[:, N_y-1] = 0
            dflux2[:, 0] = 0
            dflux2[:, 1:N_y] = np.subtract(cos_lat[:, 1:N_y]*meridion_heat_flux_time_zonal_average[:, 1:N_y], cos_lat[:, 0:N_y-1]*meridion_heat_flux_time_zonal_average[:, 0:N_y-1])
            
            term_2_vector_D = -1.*self.R_d*(dflux1/coslat_plus - dflux2/coslat_minus)/self.R_Earth/self.R_Earth/p/delta_lat/delta_lat
            print("second term of vector D, time: "+ctime())

            del dflux1, dflux2, meridion_heat_flux_time_zonal_average
            
            print(len(term_2_vector_D), len(term_2_vector_D[0]))
            
            meridion_heat_flux_yearly[m, :, :] = term_2_vector_D
            
            vector_D_cumulative = term_1_vector_D + term_2_vector_D
            print("first step of computation of streamfunction, time: "+ctime())
            
            vector_D_partial = np.zeros((dim), dtype=np.float32)
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    if k > 0 and k < (N_pressure - 1) and i > 0 and i < (N_y - 1):
                        vector_D_partial[i+k*N_y] = term_2_vector_D[k, i]
                    i = i + 1
                k = k + 1
            print("vector D2, time: "+ctime())
            
            print(term_2_vector_D[14, 70], vector_D_partial[2604])
            
            exact_solution_partial = np.dot(np.linalg.inv(linear_operator_L), vector_D_partial)
            print("exact solution for vT, time: "+ctime())
            print(exact_solution_partial)
            print(len(exact_solution_partial))
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    psi_vT[m, k, i] = exact_solution_partial[i+k*N_y]
                    i = i + 1
                k = k + 1
            print("exact solution for streamfunction for vT for year "+str(int(start_period)+m)+", time: "+ctime())
            
            dF = np.zeros((N_pressure, N_y), dtype=np.float32)
            dF[0, :] = np.subtract(10**(-4)*friction[1, :], 10**(-4)*friction[0, :])
            dF[1:N_pressure-1, :] = np.subtract(10**(-4)*friction[2:N_pressure, :], 10**(-4)*friction[0:N_pressure-2, :])
            dF[N_pressure-1, :] = np.subtract(10**(-4)*friction[N_pressure-1, :], 10**(-4)*friction[N_pressure-2, :])

            term_3_vector_D = -1.*f_Cor*dF/p_3
            print("third term of vector D, time: "+ctime())
            
            del dF, friction
            
            print(len(term_3_vector_D), len(term_3_vector_D[0]))
            
            friction_yearly[m, :, :] = term_3_vector_D
            
            vector_D_cumulative = vector_D_cumulative + term_3_vector_D
            print("second step of computation of streamfunction, time: "+ctime())
            
            vector_D_partial = np.zeros((dim), dtype=np.float32)
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    if k > 0 and k < (N_pressure - 1) and i > 0 and i < (N_y - 1):
                        vector_D_partial[i+k*N_y] = term_3_vector_D[k, i]
                    i = i + 1
                k = k + 1
            print("vector D3, time: "+ctime())
            
            print(term_3_vector_D[14, 70], vector_D_partial[2604])
            
            exact_solution_partial = np.dot(np.linalg.inv(linear_operator_L), vector_D_partial)
            print("exact solution for Fx, time: "+ctime())
            print(exact_solution_partial)
            print(len(exact_solution_partial))
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    psi_Fx[m, k, i] = exact_solution_partial[i+k*N_y]
                    i = i + 1
                k = k + 1
            print("exact solution for streamfunction for Fx for year "+str(int(start_period)+m)+", time: "+ctime())
            
            dflux = np.zeros((N_pressure, N_y), dtype=np.float32)
            dflux[0, :] = 0
            dflux[N_pressure-1, :] = 0
            dflux[:, 0] = 0
            dflux[:, N_y-1] = 0
            dflux[1:N_pressure-1, 1:N_y-1] = (cos2_forward[1:N_pressure-1, 1:N_y-1]*meridion_momentum_flux_time_zonal_average[2:N_pressure, 2:N_y]) - (cos2_forward[1:N_pressure-1, 1:N_y-1]*meridion_momentum_flux_time_zonal_average[0:N_pressure-2, 2:N_y]) - (cos2_backward[1:N_pressure-1, 1:N_y-1]*meridion_momentum_flux_time_zonal_average[2:N_pressure, 0:N_y-2]) + (cos2_backward[1:N_pressure-1, 1:N_y-1]*meridion_momentum_flux_time_zonal_average[0:N_pressure-2, 0:N_y-2])

            term_4_vector_D = f_Cor*dflux/self.R_Earth/cos2_current/lat_3/p_3
            print("fourth term of vector D, time: "+ctime())
                
            del dflux, meridion_momentum_flux_time_zonal_average
            
            print(len(term_4_vector_D), len(term_4_vector_D[0]))
            
            momentum_flux_yearly[m, :, :] = term_4_vector_D
            
            vector_D_cumulative = vector_D_cumulative + term_4_vector_D
            print("third step of computation of streamfunction, time: "+ctime())
            
            vector_D_partial = np.zeros((dim), dtype=np.float32)
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    if k > 0 and k < (N_pressure - 1) and i > 0 and i < (N_y - 1):
                        vector_D_partial[i+k*N_y] = term_4_vector_D[k, i]
                    i = i + 1
                k = k + 1
            print("vector D4, time: "+ctime())
            
            print(term_4_vector_D[14, 70], vector_D_partial[2604])
            
            exact_solution_partial = np.dot(np.linalg.inv(linear_operator_L), vector_D_partial)
            print("exact solution for uv, time: "+ctime())
            print(exact_solution_partial)
            print(len(exact_solution_partial))
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    psi_uv[m, k, i] = exact_solution_partial[i+k*N_y]
                    i = i + 1
                k = k + 1
            print("exact solution for streamfunction for uv for year "+str(int(start_period)+m)+", time: "+ctime())

            dflux1 = np.zeros((N_pressure, N_y), dtype=np.float32)
            dflux2 = np.zeros((N_pressure, N_y), dtype=np.float32)
            dflux1[0:N_pressure-1, :] = np.subtract(omega_flux_time_zonal_average[1:N_pressure, :], omega_flux_time_zonal_average[0:N_pressure-1, :])
            dflux1[N_pressure-1, :] = 0
            dflux2[0, :] = 0
            dflux2[1:N_pressure, :] = np.subtract(omega_flux_time_zonal_average[1:N_pressure, :], omega_flux_time_zonal_average[0:N_pressure-1, :])

            term_5a_vector_D = f_Cor*dflux1/p_1/p_2
            term_5b_vector_D = f_Cor*dflux2/p_2/p_4
            term_5_vector_D = term_5a_vector_D - term_5b_vector_D
            print("fifth term of vector D, time: "+ctime())
            
            del dflux1, dflux2, omega_flux_time_zonal_average, term_5a_vector_D, term_5b_vector_D

            print(len(term_5_vector_D), len(term_5_vector_D[0]))
            
            omega_flux_yearly[m, :, :] = term_5_vector_D
            
            vector_D_cumulative = vector_D_cumulative + term_5_vector_D
            print("fourth step of computation of streamfunction, time: "+ctime())
            
            vector_D_partial = np.zeros((dim), dtype=np.float32)
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    if k > 0 and k < (N_pressure - 1) and i > 0 and i < (N_y - 1):
                        vector_D_partial[i+k*N_y] = term_5_vector_D[k, i]
                    i = i + 1
                k = k + 1
            print("vector D5, time: "+ctime())
            
            print(term_5_vector_D[14, 70], vector_D_partial[2604])
            
            exact_solution_partial = np.dot(np.linalg.inv(linear_operator_L), vector_D_partial)
            print("exact solution for uw, time: "+ctime())
            print(exact_solution_partial)
            print(len(exact_solution_partial))
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    psi_uw[m, k, i] = exact_solution_partial[i+k*N_y]
                    i = i + 1
                k = k + 1
            print("exact solution for streamfunction for uw for year "+str(int(start_period)+m)+", time: "+ctime())
            
            dflux = np.zeros((N_pressure, N_y), dtype=np.float32)
            dflux[0, :] = 0
            dflux[N_pressure-1, :] = 0
            dflux[:, 0] = 0
            dflux[:, N_y-1] = 0
            dflux[1:N_pressure-1, 1:N_y-1] = theta_flux_time_zonal_average[2:N_pressure, 2:N_y] - theta_flux_time_zonal_average[0:N_pressure-2, 2:N_y] - theta_flux_time_zonal_average[2:N_pressure, 0:N_y-2] + theta_flux_time_zonal_average[0:N_pressure-2, 0:N_y-2]

            term_6_vector_D = -1.*self.R_d*dflux/self.R_Earth/((100*self.p_0)**(self.R_d/self.c_p))/p_star/lat_3/p_3
            print("sixth term of vector D, time: "+ctime())
                
            del dflux, theta_flux_time_zonal_average

            print(len(term_6_vector_D), len(term_6_vector_D[0]))
            
            theta_flux_yearly[m, :, :] = term_6_vector_D
            
            vector_D_cumulative = vector_D_cumulative + term_6_vector_D
            print("fifth step of computation of streamfunction, time: "+ctime())
            
            vector_D_partial = np.zeros((dim), dtype=np.float32)
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    if k > 0 and k < (N_pressure - 1) and i > 0 and i < (N_y - 1):
                        vector_D_partial[i+k*N_y] = term_6_vector_D[k, i]
                    i = i + 1
                k = k + 1
            print("vector D6, time: "+ctime())
            
            print(term_6_vector_D[14, 70], vector_D_partial[2604])
            
            exact_solution_partial = np.dot(np.linalg.inv(linear_operator_L), vector_D_partial)
            print("exact solution for wTh, time: "+ctime())
            print(exact_solution_partial)
            print(len(exact_solution_partial))
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    psi_wTh[m, k, i] = exact_solution_partial[i+k*N_y]
                    i = i + 1
                k = k + 1
            print("exact solution for streamfunction for wTh for year "+str(int(start_period)+m)+", time: "+ctime())
            
            del vector_D_partial, exact_solution_partial

            factor_D_cumulative = vector_D_cumulative
            print("factor D cumulative, time: "+ctime())
            
            del p_1, p_2, p_3, p_4, lat_1, lat_2, lat_3, lat_4, f_Cor, cos_lat, delta_lat, p, delta_p, p_star, cos2_forward, cos2_current, cos2_backward
            del coslat_plus, coslat_minus
            del factor_a, factor_b, factor_c, factor_d, factor_A1, factor_A2, factor_A3, factor_B1, factor_B2, factor_B3, factor_C1, factor_C2, factor_C3
            del term_1_vector_D, term_2_vector_D, term_3_vector_D, term_4_vector_D, term_5_vector_D, term_6_vector_D
            del vector_D_cumulative
            
            print(len(factor_D_cumulative), len(factor_D_cumulative[0]))
            
#            vector_x = np.zeros((dim), dtype=np.float32)
            vector_D = np.zeros((dim), dtype=np.float32)
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    if k > 0 and k < (N_pressure - 1) and i > 0 and i < (N_y - 1):
                        vector_D[i+k*N_y] = factor_D_cumulative[k, i]
                    i = i + 1
                k = k + 1
            print("vector D, time: "+ctime())
            
            print(vector_D[543:724], factor_D_cumulative[3])
            
            print(factor_D_cumulative[14, 70], vector_D[2604])
            
            del factor_D_cumulative
            
            L = scipy.sparse.csr_matrix(linear_operator_L)
            
            #exact_solution = np.dot(np.linalg.inv(linear_operator_L), vector_D)
            exact_solution = scipy.sparse.linalg.spsolve(L, vector_D)
            print("exact solution, time: "+ctime())
            print(exact_solution)
            print(len(exact_solution))
            
            streamfunction = np.zeros((N_pressure, N_y), dtype=np.float32)
            
            k = 0
            while k < N_pressure:
                i = 0
                while i < N_y:
                    streamfunction[k, i] = exact_solution[i+k*N_y]
                    i = i + 1
                k = k + 1
            print("exact solution for streamfunction, time: "+ctime())
            
            print(len(streamfunction), len(streamfunction[0]))
            print(streamfunction)
            
            streamfunction_yearly[m, :, :] = (streamfunction/(10**11))

#            vector_x[:] = 0.
#            print("vector x, time: "+ctime())
#            print(vector_x)
#            
#            weight = 1.9
#            dim = len(vector_x)
#            
#            iteration = 0
#            while iteration < self.N_iterations:
#                i = 0
#                while i < dim:
#                    s = 0
#                    j = 0
#                    while j < dim:
#                        if j != i:
#                            s = s + linear_operator_L[i, j]*vector_x[j]
#                        j = j + 1
#                    vector_x[i] = vector_x[i] + weight*(((vector_D[i] - s)/linear_operator_L[i, i]) - vector_x[i])
#                    i = i + 1
#                print(str(iteration)+", time: "+ctime())
#                iteration = iteration + 1
#                
#            streamfunction_SOR = np.zeros((N_pressure, N_y), dtype=np.float32)
#            
#            k = 0
#            while k < N_pressure:
#                i = 0
#                while i < N_y:
#                    streamfunction_SOR[k, i] = vector_x[i+k*N_y]
#                    i = i + 1
#                k = k + 1
#            print("streamfunction, time: "+ctime())
#            
#            print(len(streamfunction_SOR), len(streamfunction_SOR[0]))
#            print(streamfunction_SOR)
#            print(streamfunction_SOR[0])
#            print(streamfunction_SOR[1])
#            
#            streamfunction_SOR_yearly[m, :, :] = (streamfunction_SOR/(10**11))

            linear_operator_L_yearly[m, :, :] = linear_operator_L
            linear_operator_L_S2_yearly[m, :, :] = linear_operator_L_S2
            linear_operator_L_u_yearly[m, :, :] = linear_operator_L_u
            linear_operator_L_T_yearly[m, :, :] = linear_operator_L_T
            vector_psi_yearly[m, :] = exact_solution
            vector_D_yearly[m, :] = vector_D

            del linear_operator_L, linear_operator_L_S2, linear_operator_L_u, linear_operator_L_T, exact_solution, vector_D, L, streamfunction
            
            print("end of computation of streamfunction for year "+str(int(start_period)+m)+", time: "+ctime())
            
            m = m + 1

        np.save("{0}_linear_operator_L_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), linear_operator_L_yearly)
        np.save("{0}_linear_operator_L_S2_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), linear_operator_L_S2_yearly)
        np.save("{0}_linear_operator_L_u_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), linear_operator_L_u_yearly)
        np.save("{0}_linear_operator_L_T_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), linear_operator_L_T_yearly)
        np.save("{0}_vector_D_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), vector_D_yearly)
        np.save("{0}_vector_psi_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), vector_psi_yearly)
        np.save("{0}_streamfunction_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), streamfunction_yearly)
        np.save("{0}_diabatic_heating_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), diabatic_heating_yearly)
        np.save("{0}_heat_flux_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), meridion_heat_flux_yearly)
        np.save("{0}_friction_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), friction_yearly)
        np.save("{0}_momentum_flux_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), momentum_flux_yearly)
        np.save("{0}_omega_flux_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), omega_flux_yearly)
        np.save("{0}_theta_flux_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), theta_flux_yearly)
        np.save("{0}_term_J_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), diabatic_heating_averaged_yearly)
        np.save("{0}_term_vT_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), meridion_heat_flux_averaged_yearly)
        np.save("{0}_term_Fx_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), friction_averaged_yearly)
        np.save("{0}_term_uv_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), meridion_momentum_flux_averaged_yearly)
        np.save("{0}_term_uw_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), omega_flux_averaged_yearly)
        np.save("{0}_term_wTh_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), theta_flux_averaged_yearly)
        np.save("{0}_psi_J_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), psi_J)
        np.save("{0}_psi_vT_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), psi_vT)
        np.save("{0}_psi_Fx_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), psi_Fx)
        np.save("{0}_psi_uv_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), psi_uv)
        np.save("{0}_psi_uw_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), psi_uw)
        np.save("{0}_psi_wTh_full_KE_{1}_{2}_{3}_{4}".format(self.reanalysis, self.area, self.period, int(year[0]), int(year[-1])), psi_wTh)

        return lon, lat, pressure, year, diabatic_heating_averaged_yearly, meridion_heat_flux_averaged_yearly, friction_averaged_yearly, meridion_momentum_flux_averaged_yearly, omega_flux_averaged_yearly,\
               theta_flux_averaged_yearly, streamfunction_yearly

def contributions(reanalysis, area, period, start_year):
    t1 = time.time()

    print("Start of computations: "+ctime())

    if reanalysis == "ERA5" and start_year == "1950":
        start_period = 1950
        end_period = 2018
    elif reanalysis == "ERAI" or (reanalysis == "ERA5" and start_year == "1979"):
        start_period = 1979
        end_period = 2018
    print(start_period, end_period)

    f = KuoEliassen(reanalysis, area, period, start_year)

    lon, lat, pressure, year, diabatic_heating_time_zonal_average, meridion_heat_flux_time_zonal_average, friction_time_zonal_average, meridion_momentum_flux_time_zonal_average, omega_flux_time_zonal_average, theta_flux_time_zonal_average,\
    streamfunction = f.computation()
    
    print("End of computations: "+ctime())

    t2 = time.time()
    t = t2 - t1
    print("Computations lasted "+str(t)+" seconds or "+str(int(t/3600))+" h, "+str(int((t-3600*int(t/3600))/60))+" min and "+str(t-3600*int(t/3600)-60*int((t-3600*int(t/3600))/60))+" s.")

    return plt.show()

import sys

reanalysis = sys.argv[1]
area = sys.argv[2]
period = sys.argv[3]
start_year = sys.argv[4]

contributions(reanalysis, area, period, start_year)
