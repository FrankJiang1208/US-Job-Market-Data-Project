// Initializes the page with a default plot
function init() {
    let occupation0 = []
    let salary0 = []
    let job_freq0 = []

    for (let i = 0; i < mainoccupations.length; i++) {
        row = mainoccupations[i];
        occupation0.push(row.Occupation);
        salary0.push(row.Salary);
        job_freq0.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
    }

    let trace1 = {
        x: salary0,
        y: occupation0,
        text: job_freq0,
        type: 'bar',
        orientation: 'h',
        marker: {color: 'rgb(55, 83, 109)'},
        transforms: [{
          type: 'sort',
          target: 'x',
          order: 'ascending'
        }, {
          type: 'filter',
          target: 'x',
          operation: '>',
          value: 1
        }]
    };

    let data = [trace1]


    //  https://plotly.com/javascript/reference/layout/#layout-title
    //  https://plotly.com/javascript/reference/layout/xaxis/
    let layout0 = {
        title: {
            text : 'Salary by Industry',   
        },
        xaxis: {
            title: {
                text: 'Salary ($/yr)'
                
            }
          
        },
        margin: {
          t: 150,
          l: 600,
          r: 50,
          },
          height: 1100,
    };

    Plotly.newPlot("plot", data, layout0)
}

//  setting up all occupation lists
let occupation0 = []
let salary0 = []
let job_freq0 = []

for (let i = 0; i < mainoccupations.length; i++) {
    row = mainoccupations[i];
    occupation0.push(row.Occupation);
    salary0.push(row.Salary);
    job_freq0.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
    }

let occupation1 = []
let salary1 = []
let job_freq1 = []

for (let i = 0; i < management.length; i++) {
    row = management[i];
    occupation1.push(row.Occupation);
    salary1.push(row.Salary);
    job_freq1.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
    }

    let occupation2 = []
    let salary2 = []
    let job_freq2    = []

    for (let i = 0; i < business.length; i++) {
        row = business[i];
        occupation2.push(row.Occupation);
        salary2.push(row.Salary);
        job_freq2.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
        }

        let occupation3 = []
        let salary3 = []
        let job_freq3 = []
        
        for (let i = 0; i < computer.length; i++) {
            row = computer[i];
            occupation3.push(row.Occupation);
            salary3.push(row.Salary);
            job_freq3.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
            }

            let occupation4 = []
            let salary4 = []
            let job_freq4 = []
            
            for (let i = 0; i < engineering.length; i++) {
                row = engineering[i];
                occupation4.push(row.Occupation);
                salary4.push(row.Salary);
                job_freq4.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                }
            
                let occupation5 = []
                let salary5 = []
                let job_freq5 = []
                
                for (let i = 0; i < science.length; i++) {
                    row = science[i];
                    occupation5.push(row.Occupation);
                    salary5.push(row.Salary);
                    job_freq5.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                    }
                
                    let occupation6 = []
                    let salary6 = []
                    let job_freq6 = []
                    
                    for (let i = 0; i < social.length; i++) {
                        row = social[i];
                        occupation6.push(row.Occupation);
                        salary6.push(row.Salary);
                        job_freq6.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                        }
                    
                        let occupation7 = []
                        let salary7 = []
                        let job_freq7 = []
                        
                        for (let i = 0; i < legal.length; i++) {
                            row = legal[i];
                            occupation7.push(row.Occupation);
                            salary7.push(row.Salary);
                            job_freq7.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                            }

                            let occupation8 = []
                            let salary8 = []
                            let job_freq8 = []
                            
                            for (let i = 0; i < education.length; i++) {
                                row = education[i];
                                occupation8.push(row.Occupation);
                                salary8.push(row.Salary);
                                job_freq8.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                }

                                let occupation9 = []
                                let salary9 = []
                                let job_freq9 = []
                                
                                for (let i = 0; i < arts.length; i++) {
                                    row = arts[i];
                                    occupation9.push(row.Occupation);
                                    salary9.push(row.Salary);
                                    job_freq9.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                    }

                                    let occupation10 = []
                                    let salary10 = []
                                    let job_freq10 = []
                                    
                                    for (let i = 0; i < healthcare.length; i++) {
                                        row = healthcare[i];
                                        occupation10.push(row.Occupation);
                                        salary10.push(row.Salary);
                                        job_freq10.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                        }

                                        let occupation11 = []
                                        let salary11 = []
                                        let job_freq11 = []
                                        
                                        for (let i = 0; i < healthcaresup.length; i++) {
                                            row = healthcaresup[i];
                                            occupation11.push(row.Occupation);
                                            salary11.push(row.Salary);
                                            job_freq11.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                            }

                                            let occupation12 = []
                                            let salary12 = []
                                            let job_freq12 = []
                                            
                                            for (let i = 0; i < protective.length; i++) {
                                                row = protective[i];
                                                occupation12.push(row.Occupation);
                                                salary12.push(row.Salary);
                                                job_freq12.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                }

                                                let occupation13 = []
                                                let salary13 = []
                                                let job_freq13 = []
                                                
                                                for (let i = 0; i < food.length; i++) {
                                                    row = food[i];
                                                    occupation13.push(row.Occupation);
                                                    salary13.push(row.Salary);
                                                    job_freq13.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                    }

                                                    let occupation14 = []
                                                    let salary14 = []
                                                    let job_freq14 = []
                                                    
                                                    for (let i = 0; i < building.length; i++) {
                                                        row = building[i];
                                                        occupation14.push(row.Occupation);
                                                        salary14.push(row.Salary);
                                                        job_freq14.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                        }

                                                        let occupation15 = []
                                                        let salary15 = []
                                                        let job_freq15 = []
                                                        
                                                        for (let i = 0; i < personalcare.length; i++) {
                                                            row = personalcare[i];
                                                            occupation15.push(row.Occupation);
                                                            salary15.push(row.Salary);
                                                            job_freq15.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                            }

                                                            let occupation16 = []
                                                            let salary16 = []
                                                            let job_freq16 = []
                                                            
                                                            for (let i = 0; i < sales.length; i++) {
                                                                row = sales[i];
                                                                occupation16.push(row.Occupation);
                                                                salary16.push(row.Salary);
                                                                job_freq16.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                                }

                                                                let occupation17 = []
                                                                let salary17 = []
                                                                let job_freq17 = []
                                                                
                                                                for (let i = 0; i < office.length; i++) {
                                                                    row = office[i];
                                                                    occupation17.push(row.Occupation);
                                                                    salary17.push(row.Salary);
                                                                    job_freq17.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                                    }

                                                                    let occupation18 = []
                                                                    let salary18 = []
                                                                    let job_freq18 = []
                                                                    
                                                                    for (let i = 0; i < farming.length; i++) {
                                                                        row = farming[i];
                                                                        occupation18.push(row.Occupation);
                                                                        salary18.push(row.Salary);
                                                                        job_freq18.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                                        }

                                                                        let occupation19 = []
                                                                        let salary19 = []
                                                                        let job_freq19 = []
                                                                        
                                                                        for (let i = 0; i < construction.length; i++) {
                                                                            row = construction[i];
                                                                            occupation19.push(row.Occupation);
                                                                            salary19.push(row.Salary);
                                                                            job_freq19.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                                            }

                                                                            let occupation20 = []
                                                                            let salary20 = []
                                                                            let job_freq20 = []
                                                                            
                                                                            for (let i = 0; i < maintenance.length; i++) {
                                                                                row = maintenance[i];
                                                                                occupation20.push(row.Occupation);
                                                                                salary20.push(row.Salary);
                                                                                job_freq20.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                                                }

                                                                                let occupation21 = []
                                                                                let salary21 = []
                                                                                let job_freq21 = []
                                                                                
                                                                                for (let i = 0; i < production.length; i++) {
                                                                                    row = production[i];
                                                                                    occupation21.push(row.Occupation);
                                                                                    salary21.push(row.Salary);
                                                                                    job_freq21.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                                                    }

                                                                                    let occupation22 = []
                                                                                    let salary22 = []
                                                                                    let job_freq22 = []
                                                                                    
                                                                                    for (let i = 0; i < transportation.length; i++) {
                                                                                        row = transportation[i];
                                                                                        occupation22.push(row.Occupation);
                                                                                        salary22.push(row.Salary);
                                                                                        job_freq22.push('Job Frequency per 1000 jobs: ' + row.Job_Frequency)
                                                                                        }

// Call updatePlotly() when a change takes place to the DOM
d3.selectAll("#selDataset").on("change", updatePlotly);
// This function is called when a dropdown menu item is selected
function updatePlotly() {
    // Use D3 to select the dropdown menu
    var dropdownMenu = d3.select("#selDataset");
    // Assign the value of the dropdown menu option to a variable
    var dataset = dropdownMenu.property("value");
  
    // Initialize x and y arrays
    var occupation = [];
    var salary = [];
    var layout = {}


    if (dataset === 'mainoccupations') {
      occupation = occupation0;
      salary = salary0;
      layout = {
          title: "Salary by Industry"   
      }
    }

    else if (dataset === 'management') {
        occupation = occupation1;
        salary = salary1;
        layout = {
            title: "Management Occupations"   
        }
      }
    
    else if (dataset === 'business') {
        occupation = occupation2;
        salary = salary2;
        layout = {
            title: "Business and Financial Operations Occupations"
        }
      }
    
    else if (dataset === 'computer') {
        occupation = occupation3;
        salary = salary3;
        layout = {
            title: "Computer and Mathematical Occupations"
        }
      }

      else if (dataset === 'engineering') {
        occupation = occupation4;
        salary = salary4;
        layout = {
            title: "Architecture and Engineering Occupations"
        }
      }

      else if (dataset === 'science') {
        occupation = occupation5;
        salary = salary5;
        layout = {
            title: "Life, Physical, and Social Science Occupations"
        }
      }

      else if (dataset === 'social') {
        occupation = occupation6;
        salary = salary6;
        layout = {
            title: "Community and Social Service Occupations"
        }
      }

      else if (dataset === 'legal') {
        occupation = occupation7;
        salary = salary7;
        layout = {
            title: "Legal Occupations"
        }
      }

      else if (dataset === 'education') {
        occupation = occupation8;
        salary = salary8;
        layout = {
            title: "Educational Instruction and Library Occupations"
        }
      }

      else if (dataset === 'arts') {
        occupation = occupation9;
        salary = salary9;
        layout = {
            title: "Arts, Design, Entertainment, Sports, and Media Occupations"
        }
      }

      else if (dataset === 'healthcare') {
        occupation = occupation10;
        salary = salary10;
        layout = {
            title: "Healthcare Practitioners and Technical Occupations"
        }
      }

      else if (dataset === 'healthcaresup') {
        occupation = occupation11;
        salary = salary11;
        layout = {
            title: "Healthcare Support Occupations"
        }
      }

      else if (dataset === 'protective') {
        occupation = occupation12;
        salary = salary12;
        layout = {
            title: "Protective Service Occupations"
        }
      }

      else if (dataset === 'food') {
        occupation = occupation13;
        salary = salary13;
        layout = {
            title: "Food Preparation and Serving Related Occupations"
        }
      }

      else if (dataset === 'building') {
        occupation = occupation14;
        salary = salary14;
        layout = {
            title: "Building and Grounds Cleaning and Maintenance Occupations"
        }
      }

      else if (dataset === 'personalcare') {
        occupation = occupation15;
        salary = salary15;
        layout = {
            title: "Personal Care and Service Occupations"
        }
      }

      else if (dataset === 'sales') {
        occupation = occupation16;
        salary = salary16;
        layout = {
            title: "Sales and Related Occupations"
        }
      }

      else if (dataset === 'office') {
        occupation = occupation17;
        salary = salary17;
        layout = {
            title: "Office and Administrative Support Occupations"
        }
      }

      else if (dataset === 'farming') {
        occupation = occupation18;
        salary = salary18;
        layout = {
            title: "Farming, Fishing, and Forestry Occupations"
        }
      }

      else if (dataset === 'construction') {
        occupation = occupation19;
        salary = salary19;
        layout = {
            title: "Construction and Extraction Occupations"
        }
      }

      else if (dataset === 'maintenance') {
        occupation = occupation20;
        salary = salary20;
        layout = {
            title: "Installation, Maintenance, and Repair Occupations"
        }
      }

      else if (dataset === 'production') {
        occupation = occupation21;
        salary = salary21;
        layout = {
            title: "Production Occupations"
        }
      }

      else if (dataset === 'transportation') {
        occupation = occupation22;
        salary = salary22;
        layout = {
            title: "Transportation and Material Moving Occupations"
        }
      }


    // Note the extra brackets around 'x' and 'y'
    Plotly.restyle("plot", "x", [salary]);
    Plotly.restyle("plot", "y", [occupation]);
    Plotly.relayout("plot", layout)
}

init();
