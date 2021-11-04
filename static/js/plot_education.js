// let High_school =0
// let Bachelors=0
// let Associat=0
// let other=0
// let Vocational=0
// let None=0
// let Masters=0
// let phd=0

// for( let i=0; i<BestJobs.lenghth;i++){
//     let job= BestJobs[i]

//     if (BestJobs.education==="None") {
//         None+=1 
//       } else if (BestJobs.education==="Associate's Degree"||"2 Year Degree") {
//         Associat+=1
//       } else if (BestJobs.education==="Bachelor's Degree"||"4 Year Degree") {
//         Bachelors+=1
//       } else if (BestJobs.education==="Vocational Certificate") {
//         Vocational+=1
//       } else if (BestJobs.education==="Master's Degree"||"Graduate Degree") {
//         Masters+=1
//       }else if (BestJobs.education==="Ph.D."||"Doctorate") {
//         phd+=1
//     }else if (BestJobs.education==="Other") {
//         other+=1}
// }

// let degree =["No Degree","High School","Associate's Degree","Vocational Certificate","Bachelor's Degree","Master's Degree","Ph.D. or Doctorate","Other Degrees"]
// let degree_count=[111,3024,765,168,1551,130,25,381]

// function plotEducation(x,y){
//     let trace1 = {
//       x: x,
//       y: y,
//       type: "bar"
//     }

//     let data=[trace1]

//     let layout={
//         title:"Number of jobs availible per level of Education"
//     };

//     Plotly.newPlot("plot",data,layout);
// }




  var chart = new CanvasJS.Chart("plot3", {
    animationEnabled: true,
    title:{
      text: "Number of Jobs Available per Level of Education",
      horizontalAlign: "left"
    },
    data: [{
      type: "doughnut",
      startAngle: 240,
      //innerRadius: 60,
      indexLabelFontSize: 17,
      indexLabel: "{label} - #percent%",
      toolTipContent: "<b>{label}:</b> {y} (#percent%)",
      dataPoints: [
        { y: 111, label: "No Degree" },
        { y: 3024, label: "High School" },
        { y: 765, label: "Associate's Degree" },
        { y: 168, label: "Vocational Certificate"},
        { y: 1551, label: "Bachelor's Degree"},
        { y: 130, label: "Master's Degree"},
        { y: 130, label: "Ph.D. or Doctorate"},
        { y: 25, label: "Other Degrees"}
      
      ]
    }]
  });
  chart.render();
  
  
