### Data analysis project in chapter 2

import os
from six.moves import urllib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DOWNLOAD_ROOT = "http://s3.amazonaws.com/gamma-datasets/"
TRAFFIC_PATH = "datasets"
TRAFFIC_URL = DOWNLOAD_ROOT + "Speed_Camera_Violations.csv"


## Use following codes for fetching data

def fetch_traffic_data(traffic_url=TRAFFIC_URL, traffic_path=TRAFFIC_PATH):
    if not os.path.isdir(traffic_path):
        os.makedirs(traffic_path)
    csv_path = os.path.join(traffic_path, "Speed_Camera_Violations.csv")
    urllib.request.urlretrieve(traffic_url, csv_path)

    return pd.read_csv(csv_path, parse_dates=[2], index_col=[2])


traffic = fetch_traffic_data()
traffic["VIOLATIONS"].count()

import datetime as dt
traffic["Year Month"] = traffic.index.to_series().apply(lambda x: dt.datetime.strftime(x, '%Y%m'))
pd.to_datetime(traffic.index.values.astype(float))
traffic.plot(kind="scatter", x="LONGITUDE", y="LATITUDE", alpha=0.4, label="Traffic Violations per Case",
              figsize=(10,7),s = traffic["VIOLATIONS"], c=traffic.index, cmap=plt.get_cmap('Reds'),
              colorbar=True, sharex=False)

