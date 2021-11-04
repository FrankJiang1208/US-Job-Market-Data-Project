// Get indeed data with d3
d3.json('../static/resources/indeed_jobs_w.json').then(function(response) {
    var markers = L.markerClusterGroup();
    var keys = Object.keys(response);
  
    // Define marker icon
    var myIcon = L.icon({
      iconUrl: '../static/images/pin_map.png',
      iconSize: [38, 38],
      iconAnchor: [22, 94],
      popupAnchor: [-3, -76]
      });
  
    for(var i = 0; i < keys.length; i++){
      if (response[keys[i]].latitude){
        markers.addLayer(L.marker([response[keys[i]].latitude, response[keys[i]].longitude], {icon: myIcon}).bindPopup(`<h5><b>${response[keys[i]].job_title}</b></h5><hr>${response[keys[i]].company_name}<br>(${response[keys[i]].category})`));
      };
    };
  
    // Get USAjobs data with d3
    d3.json('../static/resources/gov_jobs.json').then(function(response) {
  
      var markers2 = L.markerClusterGroup();
  
      var keys2 = Object.keys(response);
  
      for(var i = 0; i < keys2.length; i++){
        if (response[keys2[i]].LocationLatitude){
          markers2.addLayer(L.marker([response[keys2[i]].LocationLatitude, response[keys2[i]].LocationLongitude], {icon: myIcon}).bindPopup(`<h5><b>${response[keys2[i]].Title}</b></h5><hr>${response[keys2[i]].Agency}<br>(${response[keys2[i]].Department})`));
        };
      };
  
  
      // Adding the tile layer
      // var myTileLayer = L.tileLayer('https://tiles.stadiamaps.com/tiles/osm_bright/{z}/{x}/{y}{r}.png', {
      // maxZoom: 20,
      // attribution: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
      // });
      myTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      });
  
      var baseMaps = {
        "Jobs": myTileLayer
        };
      
      // Create an overlay object
      var overlayMaps = {
        "Public & Private Sectors": markers,
        "Government": markers2
      };
  
      // Define a map object
      var myMap = L.map("map", {
        center: [35,-100],
        zoom: 5,
        layers: [myTileLayer, markers]
        });
  
      // Add the layer control to the map
      L.control.layers(baseMaps, overlayMaps, {
        collapsed: false
      }).addTo(myMap);
    });
  });
  