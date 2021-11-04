let city=["Atlanta","Chicago","Houston","New York","Dallas","Charlotte","Phoenix","Orlando","Los Angeles","Tampa","St. Louis","Jacksonville","San Diego","Memphis","Indianapolis","San Antonio","Columbus","Seattle","Cincinnati","Austin"]
let Job_count=[179,124,102,95,93,86,85,80,74,74,71,65,57,57,56,55,53,48,47,47]



function plotCity(x,y){
    let trace1 = {
      x: x,
      y: y,
      type: "bar"
    }

    let data=[trace1]

    let layout={
        title:"Top 20 Cities with Higher Number of Job Openings"
    };

    Plotly.newPlot("plot2",data,layout);
}
plotCity(city,Job_count)

