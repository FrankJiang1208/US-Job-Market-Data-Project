
geoData = "../static/resources/geojsons/datachoropleth.geojson"
   
    
  let geojson;



  d3.json(geoData).then(function(data) {
    let myMap = L.map("map", {
        center: [37.8, -96],
        zoom: 5 ,
        minZoom: 4,
        // layers: [streets, geojson]
      });
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        })
        .addTo(myMap);
    const bigboi = data.features
    keys=Object.keys(bigboi[0].properties.jobs['Code'])
    dropdown=d3.select("#selDataset");

    for (i=0;i<keys.length;i++) {
      let op=dropdown.append("option").text(keys[i])
      op.property('value',keys[i])
  }
  function init(){
    map.remove();
    div=d3.select('body').append("div").attr("id","map")
    let myMap1 = L.map("map", {
      center: [37.8, -96],
      zoom: 5 ,
      minZoom: 4,
      // layers: [streets, geojson]
    });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      })
      .addTo(myMap1);
    
    
      geoData = "../static/resources/geojsons/datachoropleth.geojson"
  
    // Create a new choropleth layer.
    geojson = L.choropleth(data, {

    // Define which property in the features to use.
    valueProperty: "Business Intelligence Developer",

    // Set the color scale.
    scale: ["#5ba3ff", "#301934"],

    // The number of breaks in the step range
    steps: 8,

    // q for quartile, e for equidistant, k for k-means
    mode: "q",
    style: {
      // Border color
      color: "#fff",
      weight: 1,
      fillOpacity: 0.8
    },

    // Binding a popup to each layer
    onEachFeature: function(feature, layer) {
      if( feature.properties["Business Intelligence Developer"]!== undefined){
          layer.bindPopup(feature.properties.name + "<br><hr>Number of Jobs: " +
          feature.properties["Business Intelligence Developer"]);
      }else{
        console.log(feature.properties["Business Intelligence Developer"])
        layer.bindPopup(feature.properties.name + "<br><hr>Number of Jobs: 0");
      }
    }
  })
  .addTo(myMap1)
    //   // Set up the legend.
    var legend = L.control({ position: "topleft" });
    legend.onAdd = function() {
      var div = L.DomUtil.create("div", "info legend");
      var limits = geojson.options.limits;
      var colors = geojson.options.colors;
      var labels = [];

 //     // Add the minimum and maximum.
      var legendInfo = "<h5>Occupation Density</h5>" +
        "<div class=\"labels\">" +
          "<div class=\"min\">" + limits[0] + "</div>" +
          "<div class=\"max\">" + limits[limits.length - 1] + "</div>" +
        "</div>";

      div.innerHTML = legendInfo;

      limits.forEach(function(limit, index) {
        labels.push("<li style=\"background-color: " + colors[index] + "\"></li>");
      });

      div.innerHTML += "<ul>" + labels.join("") + "</ul>";
      return div;
    };

 // //   Adding the legend to the map
    legend.addTo(myMap1);


    }

    init();

    d3.selectAll("#selDataset").on("change", getData);


    function getData(){
      map.remove();
      div=d3.select('body').append("div").attr("id","map")
      dropdown=d3.select("#selDataset"); 
      let myMap1 = L.map("map", {
        center: [37.8, -96],
        zoom: 5 ,
        minZoom: 4,
        // layers: [streets, geojson]
      });
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        })
        .addTo(myMap1);
      let v=dropdown.property('value');
      
        geoData = "../static/resources/geojsons/datachoropleth.geojson"

      // Create a new choropleth layer.
      geojson = L.choropleth(data, {

      // Define which property in the features to use.
      valueProperty: v,

      // Set the color scale.
      scale: ["#5ba3ff", "#301934"],

      // The number of breaks in the step range
      steps: 8,

      // q for quartile, e for equidistant, k for k-means
      mode: "q",
      style: {
        // Border color
        color: "#fff",
        weight: 1,
        fillOpacity: 0.8
      },

      // Binding a popup to each layer
      onEachFeature: function(feature, layer) {
        if( feature.properties[v]!== undefined){
            layer.bindPopup(feature.properties.name + "<br><hr>Number of Jobs: " +
            feature.properties[v]);
        }else{
          console.log(feature.properties[v])
          layer.bindPopup(feature.properties.name + "<br><hr>Number of Jobs: 0");
        }
      }
    })
    .addTo(myMap1)


  //   // Set up the legend.
  var legend = L.control({ position: 'topleft' });
     legend.onAdd = function() {
       var div = L.DomUtil.create("div", "info legend");
       var limits = geojson.options.limits;
       var colors = geojson.options.colors;
       var labels = [];

  //     // Add the minimum and maximum.
       var legendInfo = "<h5>Occupation Density</h5 >" +
         "<div class=\"labels\">" +
           "<div class=\"min\">" + limits[0] + "</div>" +
           "<div class=\"max\">" + limits[limits.length - 1] + "</div>" +
         "</div>";

       div.innerHTML = legendInfo;

       limits.forEach(function(limit, index) {
         labels.push("<li style=\"background-color: " + colors[index] + "\"></li>");
       });

       div.innerHTML += "<ul>" + labels.join("") + "</ul>";
       return div;
     };

  // //   Adding the legend to the map
     legend.addTo(myMap1);


 }
  });
 



  
