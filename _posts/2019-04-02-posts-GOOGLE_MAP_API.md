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
      var map = new google.visualization.Map(document.getElementById('map_div'));
      map.draw(data, {
        showTooltip: true,
        showInfoWindow: true,
        mapTypeId: 'satellite'
        zoom: 4,
        center: {lat: 40.696823, lng: -73.935390},
      });
    }
  </script>
</head>
<body>
  <div id="map_div" style="width: 800px; height: 600px"></div>
</body>
