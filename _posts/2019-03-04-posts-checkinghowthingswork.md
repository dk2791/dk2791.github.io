---
title: "Checking how things work for this website"
date: 2019-03-04
tags: []
excerpt: "Checking ipython integrability"
mathjax: true
classes: wide
---
To access script used for this post, check out [here](https://github.com/dk2791/dk2791.github.io/blob/master/_posts/2019-03-04-posts-checkinghowthingswork.md)


$$
\begin{equation*}
\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0
\end{vmatrix}
\end{equation*}
$$

```python
import os
from six.moves import urllib
import pandas as pd
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
```


```python
traffic = fetch_traffic_data()
```


```python
traffic.sort_index(inplace=True)
traffic.groupby(level=0).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIOLATIONS</th>
      <th>X COORDINATE</th>
      <th>Y COORDINATE</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
    </tr>
    <tr>
      <th>VIOLATION DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-07-01</th>
      <td>52.168421</td>
      <td>1.157331e+06</td>
      <td>1.895851e+06</td>
      <td>41.869966</td>
      <td>-87.697890</td>
    </tr>
    <tr>
      <th>2014-07-02</th>
      <td>43.228261</td>
      <td>1.157172e+06</td>
      <td>1.896513e+06</td>
      <td>41.871785</td>
      <td>-87.698460</td>
    </tr>
    <tr>
      <th>2014-07-03</th>
      <td>51.617021</td>
      <td>1.157533e+06</td>
      <td>1.896956e+06</td>
      <td>41.872993</td>
      <td>-87.697119</td>
    </tr>
    <tr>
      <th>2014-07-04</th>
      <td>59.596774</td>
      <td>1.159785e+06</td>
      <td>1.900856e+06</td>
      <td>41.883649</td>
      <td>-87.688748</td>
    </tr>
    <tr>
      <th>2014-07-05</th>
      <td>55.380952</td>
      <td>1.160100e+06</td>
      <td>1.900902e+06</td>
      <td>41.883770</td>
      <td>-87.687592</td>
    </tr>
    <tr>
      <th>2014-07-06</th>
      <td>59.983333</td>
      <td>1.159726e+06</td>
      <td>1.901113e+06</td>
      <td>41.884357</td>
      <td>-87.688963</td>
    </tr>
    <tr>
      <th>2014-07-07</th>
      <td>49.237113</td>
      <td>1.157365e+06</td>
      <td>1.895910e+06</td>
      <td>41.870129</td>
      <td>-87.697765</td>
    </tr>
    <tr>
      <th>2014-07-08</th>
      <td>48.316327</td>
      <td>1.157691e+06</td>
      <td>1.895995e+06</td>
      <td>41.870354</td>
      <td>-87.696563</td>
    </tr>
    <tr>
      <th>2014-07-09</th>
      <td>49.824742</td>
      <td>1.157944e+06</td>
      <td>1.895733e+06</td>
      <td>41.869632</td>
      <td>-87.695640</td>
    </tr>
    <tr>
      <th>2014-07-10</th>
      <td>47.309278</td>
      <td>1.157290e+06</td>
      <td>1.897096e+06</td>
      <td>41.873383</td>
      <td>-87.698004</td>
    </tr>
    <tr>
      <th>2014-07-11</th>
      <td>51.686747</td>
      <td>1.158252e+06</td>
      <td>1.894460e+06</td>
      <td>41.866131</td>
      <td>-87.694548</td>
    </tr>
    <tr>
      <th>2014-07-12</th>
      <td>42.203390</td>
      <td>1.160162e+06</td>
      <td>1.899432e+06</td>
      <td>41.879735</td>
      <td>-87.687409</td>
    </tr>
    <tr>
      <th>2014-07-13</th>
      <td>57.645161</td>
      <td>1.160090e+06</td>
      <td>1.900417e+06</td>
      <td>41.882440</td>
      <td>-87.687643</td>
    </tr>
    <tr>
      <th>2014-07-14</th>
      <td>46.145833</td>
      <td>1.157698e+06</td>
      <td>1.894575e+06</td>
      <td>41.866455</td>
      <td>-87.696585</td>
    </tr>
    <tr>
      <th>2014-07-15</th>
      <td>46.857143</td>
      <td>1.158040e+06</td>
      <td>1.895050e+06</td>
      <td>41.867752</td>
      <td>-87.695312</td>
    </tr>
    <tr>
      <th>2014-07-16</th>
      <td>48.646465</td>
      <td>1.158067e+06</td>
      <td>1.895417e+06</td>
      <td>41.868761</td>
      <td>-87.695204</td>
    </tr>
    <tr>
      <th>2014-07-17</th>
      <td>48.103093</td>
      <td>1.158137e+06</td>
      <td>1.895128e+06</td>
      <td>41.867964</td>
      <td>-87.694953</td>
    </tr>
    <tr>
      <th>2014-07-18</th>
      <td>52.376471</td>
      <td>1.158751e+06</td>
      <td>1.894366e+06</td>
      <td>41.865862</td>
      <td>-87.692724</td>
    </tr>
    <tr>
      <th>2014-07-19</th>
      <td>58.274194</td>
      <td>1.160721e+06</td>
      <td>1.899518e+06</td>
      <td>41.879958</td>
      <td>-87.685355</td>
    </tr>
    <tr>
      <th>2014-07-20</th>
      <td>52.904762</td>
      <td>1.160850e+06</td>
      <td>1.900023e+06</td>
      <td>41.881340</td>
      <td>-87.684866</td>
    </tr>
    <tr>
      <th>2014-07-21</th>
      <td>49.322222</td>
      <td>1.159274e+06</td>
      <td>1.895277e+06</td>
      <td>41.868352</td>
      <td>-87.690772</td>
    </tr>
    <tr>
      <th>2014-07-22</th>
      <td>50.811111</td>
      <td>1.159185e+06</td>
      <td>1.895274e+06</td>
      <td>41.868345</td>
      <td>-87.691100</td>
    </tr>
    <tr>
      <th>2014-07-23</th>
      <td>51.318681</td>
      <td>1.159495e+06</td>
      <td>1.895639e+06</td>
      <td>41.869340</td>
      <td>-87.689954</td>
    </tr>
    <tr>
      <th>2014-07-24</th>
      <td>52.321839</td>
      <td>1.159703e+06</td>
      <td>1.896104e+06</td>
      <td>41.870612</td>
      <td>-87.689177</td>
    </tr>
    <tr>
      <th>2014-07-25</th>
      <td>48.236842</td>
      <td>1.159034e+06</td>
      <td>1.896310e+06</td>
      <td>41.871189</td>
      <td>-87.691635</td>
    </tr>
    <tr>
      <th>2014-07-26</th>
      <td>56.777778</td>
      <td>1.161149e+06</td>
      <td>1.899982e+06</td>
      <td>41.881223</td>
      <td>-87.683770</td>
    </tr>
    <tr>
      <th>2014-07-27</th>
      <td>57.903226</td>
      <td>1.160853e+06</td>
      <td>1.899920e+06</td>
      <td>41.881059</td>
      <td>-87.684861</td>
    </tr>
    <tr>
      <th>2014-07-28</th>
      <td>46.666667</td>
      <td>1.159703e+06</td>
      <td>1.896104e+06</td>
      <td>41.870612</td>
      <td>-87.689177</td>
    </tr>
    <tr>
      <th>2014-07-29</th>
      <td>44.534247</td>
      <td>1.160348e+06</td>
      <td>1.898468e+06</td>
      <td>41.877085</td>
      <td>-87.686749</td>
    </tr>
    <tr>
      <th>2014-07-30</th>
      <td>47.000000</td>
      <td>1.160617e+06</td>
      <td>1.898516e+06</td>
      <td>41.877210</td>
      <td>-87.685761</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-11-24</th>
      <td>35.171053</td>
      <td>1.159490e+06</td>
      <td>1.894617e+06</td>
      <td>41.866529</td>
      <td>-87.690028</td>
    </tr>
    <tr>
      <th>2018-11-25</th>
      <td>22.191781</td>
      <td>1.158107e+06</td>
      <td>1.898191e+06</td>
      <td>41.876368</td>
      <td>-87.694995</td>
    </tr>
    <tr>
      <th>2018-11-26</th>
      <td>8.547009</td>
      <td>1.159749e+06</td>
      <td>1.893199e+06</td>
      <td>41.862637</td>
      <td>-87.689100</td>
    </tr>
    <tr>
      <th>2018-11-27</th>
      <td>14.200000</td>
      <td>1.159642e+06</td>
      <td>1.891924e+06</td>
      <td>41.859141</td>
      <td>-87.689530</td>
    </tr>
    <tr>
      <th>2018-11-28</th>
      <td>15.267606</td>
      <td>1.159012e+06</td>
      <td>1.891869e+06</td>
      <td>41.859001</td>
      <td>-87.691843</td>
    </tr>
    <tr>
      <th>2018-11-29</th>
      <td>18.236111</td>
      <td>1.158425e+06</td>
      <td>1.893154e+06</td>
      <td>41.862540</td>
      <td>-87.693959</td>
    </tr>
    <tr>
      <th>2018-11-30</th>
      <td>20.423841</td>
      <td>1.158600e+06</td>
      <td>1.892246e+06</td>
      <td>41.860047</td>
      <td>-87.693342</td>
    </tr>
    <tr>
      <th>2018-12-01</th>
      <td>20.613333</td>
      <td>1.159016e+06</td>
      <td>1.895342e+06</td>
      <td>41.868530</td>
      <td>-87.691742</td>
    </tr>
    <tr>
      <th>2018-12-02</th>
      <td>25.848101</td>
      <td>1.159554e+06</td>
      <td>1.895372e+06</td>
      <td>41.868602</td>
      <td>-87.689768</td>
    </tr>
    <tr>
      <th>2018-12-03</th>
      <td>18.751724</td>
      <td>1.158749e+06</td>
      <td>1.892040e+06</td>
      <td>41.859477</td>
      <td>-87.692801</td>
    </tr>
    <tr>
      <th>2018-12-04</th>
      <td>18.297872</td>
      <td>1.159058e+06</td>
      <td>1.892202e+06</td>
      <td>41.859915</td>
      <td>-87.691666</td>
    </tr>
    <tr>
      <th>2018-12-05</th>
      <td>17.839161</td>
      <td>1.158522e+06</td>
      <td>1.893111e+06</td>
      <td>41.862422</td>
      <td>-87.693607</td>
    </tr>
    <tr>
      <th>2018-12-06</th>
      <td>18.197183</td>
      <td>1.158819e+06</td>
      <td>1.892125e+06</td>
      <td>41.859710</td>
      <td>-87.692544</td>
    </tr>
    <tr>
      <th>2018-12-07</th>
      <td>19.897436</td>
      <td>1.159356e+06</td>
      <td>1.893968e+06</td>
      <td>41.864754</td>
      <td>-87.690526</td>
    </tr>
    <tr>
      <th>2018-12-08</th>
      <td>34.746835</td>
      <td>1.159553e+06</td>
      <td>1.895374e+06</td>
      <td>41.868605</td>
      <td>-87.689775</td>
    </tr>
    <tr>
      <th>2018-12-09</th>
      <td>32.975000</td>
      <td>1.159798e+06</td>
      <td>1.895477e+06</td>
      <td>41.868884</td>
      <td>-87.688871</td>
    </tr>
    <tr>
      <th>2018-12-10</th>
      <td>18.577586</td>
      <td>1.159191e+06</td>
      <td>1.893608e+06</td>
      <td>41.863769</td>
      <td>-87.691146</td>
    </tr>
    <tr>
      <th>2018-12-11</th>
      <td>17.264957</td>
      <td>1.159350e+06</td>
      <td>1.894265e+06</td>
      <td>41.865570</td>
      <td>-87.690540</td>
    </tr>
    <tr>
      <th>2018-12-12</th>
      <td>17.478632</td>
      <td>1.158634e+06</td>
      <td>1.894483e+06</td>
      <td>41.866183</td>
      <td>-87.693162</td>
    </tr>
    <tr>
      <th>2018-12-13</th>
      <td>16.660714</td>
      <td>1.158779e+06</td>
      <td>1.895139e+06</td>
      <td>41.867979</td>
      <td>-87.692615</td>
    </tr>
    <tr>
      <th>2018-12-14</th>
      <td>17.464912</td>
      <td>1.158872e+06</td>
      <td>1.895000e+06</td>
      <td>41.867596</td>
      <td>-87.692276</td>
    </tr>
    <tr>
      <th>2018-12-15</th>
      <td>32.986842</td>
      <td>1.159668e+06</td>
      <td>1.894936e+06</td>
      <td>41.867400</td>
      <td>-87.689364</td>
    </tr>
    <tr>
      <th>2018-12-16</th>
      <td>35.184211</td>
      <td>1.159870e+06</td>
      <td>1.894878e+06</td>
      <td>41.867237</td>
      <td>-87.688627</td>
    </tr>
    <tr>
      <th>2018-12-17</th>
      <td>19.377193</td>
      <td>1.158933e+06</td>
      <td>1.892744e+06</td>
      <td>41.861405</td>
      <td>-87.692111</td>
    </tr>
    <tr>
      <th>2018-12-18</th>
      <td>18.218487</td>
      <td>1.159433e+06</td>
      <td>1.895158e+06</td>
      <td>41.868017</td>
      <td>-87.690214</td>
    </tr>
    <tr>
      <th>2018-12-19</th>
      <td>19.854545</td>
      <td>1.159134e+06</td>
      <td>1.893288e+06</td>
      <td>41.862893</td>
      <td>-87.691366</td>
    </tr>
    <tr>
      <th>2018-12-20</th>
      <td>16.330275</td>
      <td>1.157975e+06</td>
      <td>1.894762e+06</td>
      <td>41.866962</td>
      <td>-87.695577</td>
    </tr>
    <tr>
      <th>2018-12-21</th>
      <td>18.811321</td>
      <td>1.157903e+06</td>
      <td>1.896157e+06</td>
      <td>41.870791</td>
      <td>-87.695799</td>
    </tr>
    <tr>
      <th>2018-12-22</th>
      <td>34.670886</td>
      <td>1.159684e+06</td>
      <td>1.895026e+06</td>
      <td>41.867647</td>
      <td>-87.689303</td>
    </tr>
    <tr>
      <th>2018-12-23</th>
      <td>26.040541</td>
      <td>1.158972e+06</td>
      <td>1.896172e+06</td>
      <td>41.870809</td>
      <td>-87.691882</td>
    </tr>
  </tbody>
</table>
<p>1637 rows × 5 columns</p>
</div>




```python
traffic.set_index("CAMERA ID", inplace=True)
traffic.sort_index(inplace=True)
traffic.groupby(level=0).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIOLATIONS</th>
      <th>X COORDINATE</th>
      <th>Y COORDINATE</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
    </tr>
    <tr>
      <th>CAMERA ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CHI003</th>
      <td>116.335859</td>
      <td>1.147853e+06</td>
      <td>1.934275e+06</td>
      <td>41.975605</td>
      <td>-87.731670</td>
    </tr>
    <tr>
      <th>CHI004</th>
      <td>34.781192</td>
      <td>1.148759e+06</td>
      <td>1.933818e+06</td>
      <td>41.974333</td>
      <td>-87.728347</td>
    </tr>
    <tr>
      <th>CHI005</th>
      <td>12.046569</td>
      <td>1.163048e+06</td>
      <td>1.878843e+06</td>
      <td>41.823189</td>
      <td>-87.677349</td>
    </tr>
    <tr>
      <th>CHI007</th>
      <td>59.697837</td>
      <td>1.161038e+06</td>
      <td>1.878964e+06</td>
      <td>41.823564</td>
      <td>-87.684721</td>
    </tr>
    <tr>
      <th>CHI008</th>
      <td>19.142241</td>
      <td>1.151781e+06</td>
      <td>1.898395e+06</td>
      <td>41.877071</td>
      <td>-87.718168</td>
    </tr>
    <tr>
      <th>CHI009</th>
      <td>46.374004</td>
      <td>1.151845e+06</td>
      <td>1.899805e+06</td>
      <td>41.880938</td>
      <td>-87.717898</td>
    </tr>
    <tr>
      <th>CHI010</th>
      <td>55.978475</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CHI011</th>
      <td>15.503542</td>
      <td>1.155438e+06</td>
      <td>1.906531e+06</td>
      <td>41.899325</td>
      <td>-87.704522</td>
    </tr>
    <tr>
      <th>CHI013</th>
      <td>43.584568</td>
      <td>1.182453e+06</td>
      <td>1.869712e+06</td>
      <td>41.797704</td>
      <td>-87.606445</td>
    </tr>
    <tr>
      <th>CHI014</th>
      <td>58.330670</td>
      <td>1.156276e+06</td>
      <td>1.858583e+06</td>
      <td>41.767732</td>
      <td>-87.702738</td>
    </tr>
    <tr>
      <th>CHI015</th>
      <td>13.998763</td>
      <td>1.154378e+06</td>
      <td>1.857352e+06</td>
      <td>41.764391</td>
      <td>-87.709728</td>
    </tr>
    <tr>
      <th>CHI018</th>
      <td>41.265596</td>
      <td>1.156183e+06</td>
      <td>1.859145e+06</td>
      <td>41.769276</td>
      <td>-87.703063</td>
    </tr>
    <tr>
      <th>CHI019</th>
      <td>40.185117</td>
      <td>1.156992e+06</td>
      <td>1.894531e+06</td>
      <td>41.866364</td>
      <td>-87.699143</td>
    </tr>
    <tr>
      <th>CHI020</th>
      <td>24.795818</td>
      <td>1.157040e+06</td>
      <td>1.894612e+06</td>
      <td>41.866585</td>
      <td>-87.698962</td>
    </tr>
    <tr>
      <th>CHI021</th>
      <td>116.066912</td>
      <td>1.157136e+06</td>
      <td>1.892362e+06</td>
      <td>41.860408</td>
      <td>-87.698672</td>
    </tr>
    <tr>
      <th>CHI022</th>
      <td>9.982166</td>
      <td>1.167029e+06</td>
      <td>1.830594e+06</td>
      <td>41.690702</td>
      <td>-87.664122</td>
    </tr>
    <tr>
      <th>CHI023</th>
      <td>20.836609</td>
      <td>1.166994e+06</td>
      <td>1.830711e+06</td>
      <td>41.691025</td>
      <td>-87.664248</td>
    </tr>
    <tr>
      <th>CHI024</th>
      <td>43.347826</td>
      <td>1.154183e+06</td>
      <td>1.939669e+06</td>
      <td>41.990282</td>
      <td>-87.708245</td>
    </tr>
    <tr>
      <th>CHI025</th>
      <td>17.594213</td>
      <td>1.155158e+06</td>
      <td>1.934429e+06</td>
      <td>41.975884</td>
      <td>-87.704801</td>
    </tr>
    <tr>
      <th>CHI026</th>
      <td>29.622984</td>
      <td>1.159788e+06</td>
      <td>1.923418e+06</td>
      <td>41.945574</td>
      <td>-87.688078</td>
    </tr>
    <tr>
      <th>CHI027</th>
      <td>42.721631</td>
      <td>1.159704e+06</td>
      <td>1.923559e+06</td>
      <td>41.945963</td>
      <td>-87.688384</td>
    </tr>
    <tr>
      <th>CHI028</th>
      <td>67.848101</td>
      <td>1.158541e+06</td>
      <td>1.923785e+06</td>
      <td>41.946608</td>
      <td>-87.692652</td>
    </tr>
    <tr>
      <th>CHI029</th>
      <td>77.253676</td>
      <td>1.180985e+06</td>
      <td>1.868165e+06</td>
      <td>41.793493</td>
      <td>-87.611876</td>
    </tr>
    <tr>
      <th>CHI030</th>
      <td>24.646341</td>
      <td>1.150574e+06</td>
      <td>1.871490e+06</td>
      <td>41.803265</td>
      <td>-87.723304</td>
    </tr>
    <tr>
      <th>CHI031</th>
      <td>58.457074</td>
      <td>1.150513e+06</td>
      <td>1.870811e+06</td>
      <td>41.801402</td>
      <td>-87.723545</td>
    </tr>
    <tr>
      <th>CHI032</th>
      <td>21.004773</td>
      <td>1.150881e+06</td>
      <td>1.871374e+06</td>
      <td>41.802939</td>
      <td>-87.722182</td>
    </tr>
    <tr>
      <th>CHI033</th>
      <td>20.791209</td>
      <td>1.176498e+06</td>
      <td>1.897411e+06</td>
      <td>41.873850</td>
      <td>-87.627448</td>
    </tr>
    <tr>
      <th>CHI034</th>
      <td>15.169788</td>
      <td>1.176418e+06</td>
      <td>1.897401e+06</td>
      <td>41.873824</td>
      <td>-87.627741</td>
    </tr>
    <tr>
      <th>CHI035</th>
      <td>5.924515</td>
      <td>1.178587e+06</td>
      <td>1.841967e+06</td>
      <td>41.721657</td>
      <td>-87.621463</td>
    </tr>
    <tr>
      <th>CHI036</th>
      <td>11.588529</td>
      <td>1.178616e+06</td>
      <td>1.842048e+06</td>
      <td>41.721878</td>
      <td>-87.621354</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>CHI154</th>
      <td>4.730534</td>
      <td>1.161996e+06</td>
      <td>1.836307e+06</td>
      <td>41.706487</td>
      <td>-87.682391</td>
    </tr>
    <tr>
      <th>CHI156</th>
      <td>16.366848</td>
      <td>1.132581e+06</td>
      <td>1.938004e+06</td>
      <td>41.986119</td>
      <td>-87.787741</td>
    </tr>
    <tr>
      <th>CHI157</th>
      <td>8.686924</td>
      <td>1.149395e+06</td>
      <td>1.911608e+06</td>
      <td>41.913375</td>
      <td>-87.726589</td>
    </tr>
    <tr>
      <th>CHI158</th>
      <td>29.301351</td>
      <td>1.131748e+06</td>
      <td>1.936860e+06</td>
      <td>41.982995</td>
      <td>-87.790832</td>
    </tr>
    <tr>
      <th>CHI159</th>
      <td>12.723206</td>
      <td>1.148867e+06</td>
      <td>1.910282e+06</td>
      <td>41.909746</td>
      <td>-87.728563</td>
    </tr>
    <tr>
      <th>CHI160</th>
      <td>30.117093</td>
      <td>1.149437e+06</td>
      <td>1.894438e+06</td>
      <td>41.866257</td>
      <td>-87.726879</td>
    </tr>
    <tr>
      <th>CHI161</th>
      <td>12.437132</td>
      <td>1.149006e+06</td>
      <td>1.910365e+06</td>
      <td>41.909972</td>
      <td>-87.728050</td>
    </tr>
    <tr>
      <th>CHI162</th>
      <td>18.089552</td>
      <td>1.149923e+06</td>
      <td>1.894858e+06</td>
      <td>41.867402</td>
      <td>-87.725084</td>
    </tr>
    <tr>
      <th>CHI163</th>
      <td>20.076613</td>
      <td>1.149841e+06</td>
      <td>1.894931e+06</td>
      <td>41.867603</td>
      <td>-87.725383</td>
    </tr>
    <tr>
      <th>CHI164</th>
      <td>6.827181</td>
      <td>1.155954e+06</td>
      <td>1.867614e+06</td>
      <td>41.792520</td>
      <td>-87.703676</td>
    </tr>
    <tr>
      <th>CHI165</th>
      <td>2.479060</td>
      <td>1.161996e+06</td>
      <td>1.836227e+06</td>
      <td>41.706268</td>
      <td>-87.682391</td>
    </tr>
    <tr>
      <th>CHI166</th>
      <td>1.718563</td>
      <td>1.155768e+06</td>
      <td>1.867930e+06</td>
      <td>41.793392</td>
      <td>-87.704348</td>
    </tr>
    <tr>
      <th>CHI167</th>
      <td>2.097500</td>
      <td>1.155816e+06</td>
      <td>1.868012e+06</td>
      <td>41.793615</td>
      <td>-87.704170</td>
    </tr>
    <tr>
      <th>CHI168</th>
      <td>26.908591</td>
      <td>1.167170e+06</td>
      <td>1.849194e+06</td>
      <td>41.741742</td>
      <td>-87.663073</td>
    </tr>
    <tr>
      <th>CHI169</th>
      <td>32.552524</td>
      <td>1.167083e+06</td>
      <td>1.849460e+06</td>
      <td>41.742474</td>
      <td>-87.663385</td>
    </tr>
    <tr>
      <th>CHI170</th>
      <td>6.366040</td>
      <td>1.167695e+06</td>
      <td>1.849654e+06</td>
      <td>41.742993</td>
      <td>-87.661138</td>
    </tr>
    <tr>
      <th>CHI171</th>
      <td>4.818966</td>
      <td>1.165145e+06</td>
      <td>1.920791e+06</td>
      <td>41.938253</td>
      <td>-87.668463</td>
    </tr>
    <tr>
      <th>CHI172</th>
      <td>14.719225</td>
      <td>1.165058e+06</td>
      <td>1.920991e+06</td>
      <td>41.938805</td>
      <td>-87.668779</td>
    </tr>
    <tr>
      <th>CHI173</th>
      <td>40.184932</td>
      <td>1.161334e+06</td>
      <td>1.871365e+06</td>
      <td>41.802706</td>
      <td>-87.683845</td>
    </tr>
    <tr>
      <th>CHI174</th>
      <td>114.339130</td>
      <td>1.121895e+06</td>
      <td>1.922706e+06</td>
      <td>41.944319</td>
      <td>-87.827378</td>
    </tr>
    <tr>
      <th>CHI175</th>
      <td>8.948276</td>
      <td>1.121638e+06</td>
      <td>1.922907e+06</td>
      <td>41.944874</td>
      <td>-87.828317</td>
    </tr>
    <tr>
      <th>CHI176</th>
      <td>11.000000</td>
      <td>1.122009e+06</td>
      <td>1.923002e+06</td>
      <td>41.945129</td>
      <td>-87.826952</td>
    </tr>
    <tr>
      <th>CHI177</th>
      <td>2.961538</td>
      <td>1.149783e+06</td>
      <td>1.918290e+06</td>
      <td>41.931705</td>
      <td>-87.724990</td>
    </tr>
    <tr>
      <th>CHI178</th>
      <td>14.756098</td>
      <td>1.148258e+06</td>
      <td>1.873075e+06</td>
      <td>41.807658</td>
      <td>-87.731755</td>
    </tr>
    <tr>
      <th>CHI179</th>
      <td>5.363636</td>
      <td>1.148661e+06</td>
      <td>1.873165e+06</td>
      <td>41.807896</td>
      <td>-87.730276</td>
    </tr>
    <tr>
      <th>CHI180</th>
      <td>29.145833</td>
      <td>1.183268e+06</td>
      <td>1.857719e+06</td>
      <td>41.764775</td>
      <td>-87.603828</td>
    </tr>
    <tr>
      <th>CHI181</th>
      <td>27.836735</td>
      <td>1.182997e+06</td>
      <td>1.858058e+06</td>
      <td>41.765711</td>
      <td>-87.604810</td>
    </tr>
    <tr>
      <th>CHI182</th>
      <td>58.250000</td>
      <td>1.166307e+06</td>
      <td>1.908050e+06</td>
      <td>41.903268</td>
      <td>-87.664557</td>
    </tr>
    <tr>
      <th>CHI183</th>
      <td>24.083333</td>
      <td>1.166381e+06</td>
      <td>1.908132e+06</td>
      <td>41.903490</td>
      <td>-87.664285</td>
    </tr>
    <tr>
      <th>CHI184</th>
      <td>74.446809</td>
      <td>1.182951e+06</td>
      <td>1.857908e+06</td>
      <td>41.765301</td>
      <td>-87.604985</td>
    </tr>
  </tbody>
</table>
<p>162 rows × 5 columns</p>
</div>




```python
traffic2 = fetch_traffic_data()
```


```python
import datetime as dt
traffic2.sort_index(inplace=True)
# for future reference: below gives month and year
# traffic.index.to_series().apply(lambda x: dt.datetime.strftime(x, '%b %Y'))
# A: weekday, B: month, d: date, Y: year, m: month

traffic2["weekday"] = traffic2.index.to_series().apply(lambda x: dt.datetime.strftime(x, '%A'))
traffic2.groupby(['weekday']).sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIOLATIONS</th>
      <th>X COORDINATE</th>
      <th>Y COORDINATE</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
    </tr>
    <tr>
      <th>weekday</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Friday</th>
      <td>753468</td>
      <td>2.969780e+10</td>
      <td>4.849572e+10</td>
      <td>1.072352e+06</td>
      <td>-2.246282e+06</td>
    </tr>
    <tr>
      <th>Monday</th>
      <td>686724</td>
      <td>2.976620e+10</td>
      <td>4.860353e+10</td>
      <td>1.074738e+06</td>
      <td>-2.251272e+06</td>
    </tr>
    <tr>
      <th>Saturday</th>
      <td>676715</td>
      <td>1.893861e+10</td>
      <td>3.090150e+10</td>
      <td>6.827700e+05</td>
      <td>-1.429940e+06</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>663801</td>
      <td>1.892335e+10</td>
      <td>3.087836e+10</td>
      <td>6.822298e+05</td>
      <td>-1.428801e+06</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>730603</td>
      <td>3.063207e+10</td>
      <td>5.002385e+10</td>
      <td>1.106136e+06</td>
      <td>-2.317053e+06</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>708276</td>
      <td>3.147777e+10</td>
      <td>5.140274e+10</td>
      <td>1.136724e+06</td>
      <td>-2.381161e+06</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>705136</td>
      <td>3.070936e+10</td>
      <td>5.014615e+10</td>
      <td>1.108929e+06</td>
      <td>-2.322930e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Did the number of unique camera increase? it seems yes by a marginal number
traffic2["Year"] = traffic2.index.to_series().apply(lambda x: dt.datetime.strftime(x, '%Y'))
traffic2.groupby(['Year']).agg('nunique')["CAMERA ID"].sort_index #['min', 'max', 'count', 'nunique'] possible
```




    <bound method Series.sort_index of Year
    2014    143
    2015    150
    2016    150
    2017    150
    2018    162
    Name: CAMERA ID, dtype: int64>




```python
# Any interesting pattern? Let's find out
traffic2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 170521 entries, 2014-07-01 to 2018-12-23
    Data columns (total 10 columns):
    ADDRESS         170521 non-null object
    CAMERA ID       170521 non-null object
    VIOLATIONS      170521 non-null int64
    X COORDINATE    163959 non-null float64
    Y COORDINATE    163959 non-null float64
    LATITUDE        163959 non-null float64
    LONGITUDE       163959 non-null float64
    LOCATION        163959 non-null object
    weekday         170521 non-null object
    Year            170521 non-null object
    dtypes: float64(4), int64(1), object(5)
    memory usage: 19.3+ MB



```python
traffic2.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIOLATIONS</th>
      <th>X COORDINATE</th>
      <th>Y COORDINATE</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>170521.000000</td>
      <td>1.639590e+05</td>
      <td>1.639590e+05</td>
      <td>163959.000000</td>
      <td>163959.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.880449</td>
      <td>1.159712e+06</td>
      <td>1.893473e+06</td>
      <td>41.863389</td>
      <td>-87.689229</td>
    </tr>
    <tr>
      <th>std</th>
      <td>36.636151</td>
      <td>1.570560e+04</td>
      <td>3.212393e+04</td>
      <td>0.088338</td>
      <td>0.057135</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.121638e+06</td>
      <td>1.820629e+06</td>
      <td>41.663174</td>
      <td>-87.828317</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.000000</td>
      <td>1.149783e+06</td>
      <td>1.868078e+06</td>
      <td>41.793493</td>
      <td>-87.725084</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16.000000</td>
      <td>1.159169e+06</td>
      <td>1.898488e+06</td>
      <td>41.877243</td>
      <td>-87.689803</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>37.000000</td>
      <td>1.167170e+06</td>
      <td>1.920991e+06</td>
      <td>41.939040</td>
      <td>-87.662810</td>
    </tr>
    <tr>
      <th>max</th>
      <td>479.000000</td>
      <td>1.203645e+06</td>
      <td>1.943342e+06</td>
      <td>42.000260</td>
      <td>-87.529848</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
location_plot = traffic2['VIOLATIONS'].hist(bins=20)
location_plot.set_title("Frequency and Number of Violations")
location_plot.set_xlabel("Number of Violations per Case")
location_plot.set_ylabel("Frequency of Cases")
```




    Text(0,0.5,'Frequency of Cases')


<figure style="width: 400px" class="align-center">
<img src="{{ site.url }}{{ site.baseurl }}/images/initial_test/output_10_1.png" alt="">
</figure>



```python
traffic2["Year"] = traffic2.index.to_series().apply(lambda x: dt.datetime.strftime(x, '%Y'))
traffic2.groupby(['Year']).sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIOLATIONS</th>
      <th>X COORDINATE</th>
      <th>Y COORDINATE</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014</th>
      <td>659424</td>
      <td>1.892601e+10</td>
      <td>3.093007e+10</td>
      <td>6.832156e+05</td>
      <td>-1.430905e+06</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>1201698</td>
      <td>4.221439e+10</td>
      <td>6.889669e+10</td>
      <td>1.523463e+06</td>
      <td>-3.191149e+06</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>1116805</td>
      <td>4.352645e+10</td>
      <td>7.105770e+10</td>
      <td>1.571194e+06</td>
      <td>-3.291152e+06</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>1015490</td>
      <td>4.276128e+10</td>
      <td>6.981971e+10</td>
      <td>1.543780e+06</td>
      <td>-3.233736e+06</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>931306</td>
      <td>4.271703e+10</td>
      <td>6.974767e+10</td>
      <td>1.542226e+06</td>
      <td>-3.230496e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.to_datetime(traffic2.index.values.astype(float))
traffic2.plot(kind="scatter", x="LONGITUDE", y="LATITUDE", alpha=0.4, label="Traffic Violations per Case",
              figsize=(10,7),s = traffic2["VIOLATIONS"], c=traffic2.index, cmap=plt.get_cmap('Reds'),
              colorbar=True, sharex=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x117716828>


<figure style="width: 500px" class="align-center">
<img src="{{ site.url }}{{ site.baseurl }}/images/initial_test/output_12_1.png" alt="">
<figcaption>Darker color represents newer cases, larger radius represents more violation per case.</figcaption>
</figure>


# H1 Heading

## H2 Heading

### H3 Heading

basic text.

here's *italics*

here's **bold**

to get link: [link](https://github.com/dk2791)

Bulleted lists:
* First item
+ Second item
- Third item

Here's a numbered list:
1. first
2. second
3. third

simple python block:
```python
    import numpy as np

    def sumup(x,y):
      return np.sum(x,y)
```


R code block:
```r
    library(tidyverse)
    df <- read_csv("some_file.csv")
    head(df)
```

inline code 'x+y'.
math: $$z=x+y$$
![alt]({{ site.url}}{{ site.baseurl }}/images/image3.jpeg)

good to go.
