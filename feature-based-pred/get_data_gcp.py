#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Capstone - Get Data from GCP

Created on Sat Nov  7 23:08:11 2020

@author: xuanlin
"""

# Variables ----------------------------

host_name = '130.211.203.107'
user = 'root'
password = 'Divvy200'
database = 'divvy'


path = '/Users/xuanlin/Desktop/Capstone/Optimization/data/'
trip_file_to_save = '2018Q12_top25station_trips.csv'
sensor_file_to_save = '2018Q12_top25station_sensor.csv'

# --------------------------------------


import pymysql
import pandas as pd
pd.set_option('display.max_rows', 100)

db = pymysql.connect(host_name, user, password, database)
cursor = db.cursor()
cursor.execute('USE {};'.format(database))


# function to fetch data
def fetch_sql(query, col_name=None):
    cursor = db.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    if col_name:
        return pd.DataFrame(rows, columns = col_name)
    else:
        return pd.DataFrame(rows)


# 01. Monthly Station In-flow & Out-flows ------------------------------------

query = " \
select \
    station_id, year, month, \
    sum(case when direction = 'out' then trips else 0 end) as out_trips, \
    sum(case when direction = 'in' then trips else 0 end) as in_trips \
from ( \
    select \
        from_station_id as station_id, year(start_time) as year, month(start_time) as month, \
        'out' as direction, count(*) as trips \
    from trips_dw \
    group by  \
        from_station_id, year(start_time), month(start_time), 'out' \
    union all \
    select  \
        to_station_id as station_id, year(end_time) as year, month(end_time) as month, \
        'in' as direction, count(*) as trips \
    from trips_dw \
    group by  \
        to_station_id, year(end_time), month(end_time), 'in' \
) a \
group by station_id, year, month \
;"

#top_stations = fetch_sql(query, ['station_id', 'year', 'month', 'out_trips', 'in_trips'])
#top_stations.to_csv(path + 'station_monthly_bidirect_flows.csv', index = False)

top_stations = pd.read_csv(path + 'station_monthly_bidirect_flows.csv')
top_stations['total_trips'] = top_stations['in_trips'] + top_stations['out_trips']

total_trips_6_months = top_stations[(top_stations['year'] == 2018) 
                                    & (top_stations['month'] <= 6)
                                    & (top_stations['month'] >= 1)] \
                       .groupby('station_id')['total_trips'].sum().reset_index() \
                       .sort_values('total_trips', ascending = False)
    
top_station_ids = total_trips_6_months['station_id'][:25].values

# array([ 35, 192,  91,  77,  43, 133, 174,  81,  76,  90, 177, 287, 268,
#       195,  85, 283, 100,  66, 110,  52, 181,  48,  59, 176,  49])



# 02. Extract top 25 stations ------------------------------------------------

trip_headers = list(fetch_sql('describe trips_dw;').iloc[:,0])
get_trips_top_25 = "select * \
                    from trips_dw \
                    where substr(start_time, 1, 7) between '{}' and '{}' \
                      and (from_station_id in ({}) or to_station_id in ({}))\
                    ;"
trips = fetch_sql(get_trips_top_25.format('2018-01', '2018-06',
                                          ','.join(map(str, top_station_ids)),
                                          ','.join(map(str, top_station_ids))
                                          ),
                  col_name = headers)
trips.to_csv(path + trip_file_to_save, index = False)



# 03. Extract top 20 Sensor Data ---------------------------------------------

sensor_headers = list(fetch_sql('describe station_historicals_18;').iloc[:,0])
get_trips_top_25 = "select *, str_to_date(ts_clean, '%m/%d/%Y %h:%i:%s') as date_clean\
                    from station_historicals_18 \
                    where substr(str_to_date(ts_clean, '%m/%d/%Y %h:%i:%s'), 1, 7) between '{}' and '{}' \
                      and id in ({}) \
                    ;"
sensor_data = fetch_sql(get_trips_top_25.format('2018-01', '2018-06',
                                          ','.join(map(str, top_station_ids))
                                          ),
                  col_name = headers + ['date_clean'])
sensor_data.to_csv(path + sensor_file_to_save, index = False)


# debug
sensor_data.groupby('date_clean')['id'].count().reset_index().groupby('id').count()
sensor_data[(sensor_data['id'] == 35) & (pd.Series(map(str, sensor_data['date_clean'])).str.slice(start=0,stop=10) == '2018-01-01')].count()


# need investigation and data cleaning; recorded every 10 minutes, should have 144 entries per day
# daily entries, count
# 25         344
# 50       12624
# 75           )
# some of the 


pd.Series(map(str, sensor_data['date_clean'][:5])).str.slice(start=0,stop=7)


sensor_data['date_clean'][0]




