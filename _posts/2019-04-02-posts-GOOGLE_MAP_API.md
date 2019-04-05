---
title: "Google Map API"
date: 2019-04-04
tags: []
excerpt:
mathjax: true
classes: wide
---

Top5 stations

<head>
  <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
  <script type="text/javascript">
    google.charts.load("current", {
      "packages":["map"],
      "mapsApiKey": "AIzaSyAB-pv1qhTq8z2GnUDOK9vJQyErovz2eEo"
  });
    google.charts.setOnLoadCallback(drawChart);
    function drawChart() {
      var data = google.visualization.arrayToDataTable([
        ['Lat', 'Long', 'Name'],
        [40.696823, -73.935390, 'Top 1'],
        [40.668947, -73.931834, 'Top 2'],
        [40.662563, -73.908905, 'Top 3'],
        [40.678914, -73.903900, 'Top 4'],
        [40.675401, -73.871903, 'Top 5']
      ]);
      var map = new google.visualization.Map(document.getElementById('map_div'), {
        zoom: 3,
        center: {lat: 40.696823, lng: -73.935390}
        });
      map.draw(data, {
        showTooltip: true,
        showInfoWindow: true,
        mapTypeId: 'satellite'
      });
    }
  </script>
</head>
<body>
  <div id="map_div" style="width: 800px; height: 600px"></div>
</body>

Directions:

<head>
  <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
  <meta charset="utf-8">
  <title>Displaying Text Directions With setPanel()</title>
  <style>
    #map {
      height: 425px;
    }
    html, body {
      height: 100%;
      margin: 0;
      padding: 0;
    }
    #floating-panel {
      position: absolute;
      top: 10px;
      left: 25%;
      z-index: 5;
      background-color: #fff;
      padding: 5px;
      border: 1px solid #999;
      text-align: center;
      font-family: 'Roboto','sans-serif';
      line-height: 30px;
      padding-left: 10px;
    }
    #right-panel {
      font-family: 'Roboto','sans-serif';
      line-height: 30px;
      padding-left: 10px;
    }

    #right-panel select, #right-panel input {
      font-size: 15px;
    }

    #right-panel select {
      width: 100%;
    }

    #right-panel i {
      font-size: 12px;
    }
    #right-panel {
      height: 100%;
      float: right;
      width: 390px;
      overflow: auto;
    }
    #map {
      margin-right: 400px;
    }
    #floating-panel {
      background: #fff;
      padding: 5px;
      font-size: 14px;
      font-family: Arial;
      border: 1px solid #ccc;
      box-shadow: 0 2px 2px rgba(33, 33, 33, 0.4);
      display: none;
    }
    @media print {
      #map {
        height: 500px;
        margin: 0;
      }
      #right-panel {
        float: none;
        width: auto;
      }
    }
  </style>
</head>
<body>
  <div id="floating-panel">
    <strong>Start:</strong>
    <select id="start">
      <option value="chicago, il">Chicago</option>
      <option value="st louis, mo">St Louis</option>
      <option value="joplin, mo">Joplin, MO</option>
      <option value="oklahoma city, ok">Oklahoma City</option>
      <option value="amarillo, tx">Amarillo</option>
      <option value="gallup, nm">Gallup, NM</option>
      <option value="flagstaff, az">Flagstaff, AZ</option>
      <option value="winona, az">Winona</option>
      <option value="kingman, az">Kingman</option>
      <option value="barstow, ca">Barstow</option>
      <option value="san bernardino, ca">San Bernardino</option>
      <option value="los angeles, ca">Los Angeles</option>
    </select>
    <br>
    <strong>End:</strong>
    <select id="end">
      <option value="chicago, il">Chicago</option>
      <option value="st louis, mo">St Louis</option>
      <option value="joplin, mo">Joplin, MO</option>
      <option value="oklahoma city, ok">Oklahoma City</option>
      <option value="amarillo, tx">Amarillo</option>
      <option value="gallup, nm">Gallup, NM</option>
      <option value="flagstaff, az">Flagstaff, AZ</option>
      <option value="winona, az">Winona</option>
      <option value="kingman, az">Kingman</option>
      <option value="barstow, ca">Barstow</option>
      <option value="san bernardino, ca">San Bernardino</option>
      <option value="los angeles, ca">Los Angeles</option>
    </select>
  </div>
  <div id="right-panel"></div>
  <div id="map"></div>
  <script>
    function initMap() {
      var directionsDisplay = new google.maps.DirectionsRenderer;
      var directionsService = new google.maps.DirectionsService;
      var map = new google.maps.Map(document.getElementById('map'), {
        zoom: 7,
        center: {lat: 41.85, lng: -87.65}
      });
      directionsDisplay.setMap(map);
      directionsDisplay.setPanel(document.getElementById('right-panel'));

      var control = document.getElementById('floating-panel');
      control.style.display = 'block';
      map.controls[google.maps.ControlPosition.TOP_CENTER].push(control);

      var onChangeHandler = function() {
        calculateAndDisplayRoute(directionsService, directionsDisplay);
      };
      document.getElementById('start').addEventListener('change', onChangeHandler);
      document.getElementById('end').addEventListener('change', onChangeHandler);
    }

    function calculateAndDisplayRoute(directionsService, directionsDisplay) {
      var start = document.getElementById('start').value;
      var end = document.getElementById('end').value;
      directionsService.route({
        origin: start,
        destination: end,
        travelMode: 'DRIVING'
      }, function(response, status) {
        if (status === 'OK') {
          directionsDisplay.setDirections(response);
        } else {
          window.alert('Directions request failed due to ' + status);
        }
      });
    }
  </script>
  <script async defer
  src="https://maps.googleapis.com/maps/api/js?key=AIzaSyAB-pv1qhTq8z2GnUDOK9vJQyErovz2eEo&callback=initMap">
  </script>
</body>
