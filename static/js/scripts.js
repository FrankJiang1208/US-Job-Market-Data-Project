let trace10 = {
    x: major_data.major,
    y: major_data.major_sum,
    type: "bar"
  };
let trace20 = {
    x: major_data.major,
    y: major_data.job_titles,
    type: "bar"
  };
// Data trace array
let traceData10 = [trace10,trace20];

// Apply the group barmode to the layout
let layout10 = {
  title: "Job vs Major",
  barmode: 'group'
};

// Render the plot to the div tag with id "plot"
Plotly.newPlot("plot", traceData10, layout10);

let trace30={
  x:comp_data.job_num,
  y: comp_data.major_num,
  type:'scatter'
}

let traceData20 = [trace30];

let layout20 = {
  title: "Job vs Major",
};

Plotly.newPlot("plot4", traceData20, layout20);
