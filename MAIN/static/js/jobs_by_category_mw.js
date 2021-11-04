// Using animated CanvasJS for the visual
function plotCanvas(data_canvas, main_title, y_title){

    var chart = new CanvasJS.Chart("bar", {
        theme: "light2", // "light1", "dark2", "dark1"
        backgroundColor: "#EFEFED",
        animationEnabled: true,
        animationDuration: 6000,
        height: 822,
        title: {
            padding: 15,
            text: main_title,
            fontColor: '#E37613'
        },
        axisX: {
        margin: 10,
        labelPlacement: "outside",
        labelFontSize:17,
        labelFontColor:"black",
        interval: 1,
        lineDashType: "dot"
        },
        axisY2: {
        title: y_title,
        titleFontSize: 20,
        includeZero: true,
        labelFontSize:16,
        labelFontColor:"black",
        lineThickness:3,
        margin:10
        },
        data: [{
        type: "bar",
        axisYType: "secondary",
        indexLabel: "{y}",
        dataPoints: data_canvas
        }]
    });
    chart.render();
};

function init(){
    // Indeed Jobs
    d3.json('../static/resources/indeed_jobs_w.json').then(function(response) {
        var keys = Object.keys(response);

        var category_list = []

        keys.forEach(function(key){
            var category = response[key].category;

            if (category_list.length == 0){
                category_list.push(category);
            }
            else {
            var check = 0
            category_list.forEach(function(dep){
                if(dep == category){
                check +=1
                };
            });

            if (check == 0){
                category_list.push(category);
            };
            check = 0;
            };
        });

        category_job_count_list = [];

        category_list.forEach(function(dep){
            count = 0;
            keys.forEach(function(key){
                if(dep == response[key].category){
                    count += 1;
                };
            });
            category_job_count_list.push(count);

        });

        // Create two-dimensional array (Department names & Number of job per department)
        var jobs_X = [];
        for (var i = 0; i < category_list.length; i++){
            jobs_X.push([category_list[i],category_job_count_list[i]])
        };

        jobs_X.sort((a,b) => a[1]-b[1]);

        data_canvas = [];
        jobs_X.forEach(function(element){
            data_canvas.push({label: element[0], y: element[1], color: '#3C557F', indexLabelFontSize:16, indexLabelFontColor:'#3C557F'});
        });

        plotCanvas(data_canvas, "Private & Public Sectors", "Amount of jobs per category");

        var targetDiv = document.getElementById('sample-metadata');
        targetDiv.innerHTML = "<mark class='green'><b>PROS:</b></mark><ul><li><b>Flexibility:</b> More opportunities to change jobs in order to find what suits you best.</li><li><b>More Jobs:</b> Due to the number of businesses that make up the public/private sector, there are far more opportunities available.</li><li><b>Better Progression:</b> Easier to receive a pay rise and progress within a company.</li><li><b>Better Pay:</b> Salaries are more competitive and negotiable than in government.</li></ul><hr style='border:0.5px solid #a9a9a9'><mark class='red'><b>CONS:</b></mark><ul><li><b>Benefits:</b> Benefits depend on the generosity of your employer.</li><li><b>Big Competition:</b> A lot of competition from the time you look for a job and once you’re employed. Must keep in mind two things: 1) constantly thrive for being the best in your field, and 2) be aware of the likelihood of losing your job.</li><li><b>Possible Stressful Environment:</b> You’ll need to have the ability to deal with obstacles quickly and effectively.</li></ul>"

    });
}

// Government Jobs 
d3.selectAll("#selDataset").on("change", updatePage);

function updatePage(){
    var dropdownMenu = d3.select("#selDataset");
    var dataset = dropdownMenu.property("value");

    if(dataset == "Government"){
        d3.json('../static/resources/gov_jobs.json').then(function(response) {
            var keys = Object.keys(response);
    
            var department_list = []
    
            keys.forEach(function(key){
                var department = response[key].Department;
    
                if (department_list.length == 0){
                    department_list.push(department);
                }
                else {
                var check = 0
                department_list.forEach(function(dep){
                    if(dep == department){
                    check +=1
                    };
                });
                if (check == 0){
                    department_list.push(department);
                };
                check = 0;
                };
            });
    
            department_job_count_list = [];
            minimum_wage_list = [];
            maximum_wage_list = [];
    
            var wage = 0;
    
            department_list.forEach(function(dep){
                count = 0;
                minimum = 1000000;
                maximum = 0;
                keys.forEach(function(key){
                if(dep == response[key].Department){
                    count += 1;
                    wage = parseFloat(response[key].MinimumRange);
                    if (wage != 0 && wage > 10000){
                        if (wage>maximum){maximum = wage};  
                        if (wage<minimum){minimum = wage};
                    };
                };
                });
                department_job_count_list.push(count);
                minimum_wage_list.push(minimum);
                maximum_wage_list.push(maximum);
            });
    
            // Create two-dimensional array (Department names & Number of jobs per department)
            var jobs_X = [];
            for (var i = 0; i < department_list.length; i++){
                jobs_X.push([department_list[i],department_job_count_list[i]])
            };
    
            jobs_X.sort((a,b) => a[1]-b[1]);

            data_canvas = [];
            jobs_X.forEach(function(element){
                data_canvas.push({label: element[0], y: element[1], color: '#3C557F', indexLabelFontSize:16, indexLabelFontColor:'#3C557F'});
            });
            
            plotCanvas(data_canvas, "Government", "Amount of jobs per department");

            var targetDiv = document.getElementById('sample-metadata');
            targetDiv.innerHTML = "<mark class='green'><b>PROS:</b></mark><ul><li><b>Stability:</b> The government never goes out of business.</li><li><b>Schedules:</b> Workers can easily maintain a healthy work-life balance.</li><li><b>Benefits:</b> Government benefits almost always exceed private sector benefits packages.</li><li><b>Time Off:</b> Leave time accruals are generous.</li><li><b>Jobs for Every Interest:</b> Whether your skills are accounting or engineering, or you have an interest in art, history, zoology or some other field, you can find opportunities.</li></ul><hr style='border:0.5px solid #a9a9a9'><mark class='red'><b>CONS:</b></mark><ul><li><b>Slow Salary Growth:</b> Raises are given to only a small percentage of top performers.</li><li><b>Capped earning potential:</b> Government executives are paid far less than their private-sector counterparts.</li><li><b>Low Levels of Control:</b> All major and many minor decisions to go through some sort of formal approval process.</li></ul>"          
        }); 
    }
    else if (dataset == "Private Sector"){
        d3.json('../static/resources/indeed_jobs_w.json').then(function(response) {
            var keys = Object.keys(response);
    
            var category_list = []
    
            keys.forEach(function(key){
                var category = response[key].category;   
                if (category_list.length == 0){
                    category_list.push(category);
                }
                else {
                var check = 0
                category_list.forEach(function(dep){
                    if(dep == category){
                    check +=1
                    };
                });   
                if (check == 0){
                    category_list.push(category);
                };
                check = 0;
                };
            });
    
            category_job_count_list = [];
    
            category_list.forEach(function(dep){
                count = 0;
                keys.forEach(function(key){
                    if(dep == response[key].category){
                        count += 1;
                    };
                });
                category_job_count_list.push(count);
    
            });
    
            // Create two-dimensional array (Job category name & Number of jobs per category)
            var jobs_X = [];
            for (var i = 0; i < category_list.length; i++){
                jobs_X.push([category_list[i],category_job_count_list[i]])
            };
    
            jobs_X.sort((a,b) => a[1]-b[1]);
    
            data_canvas = [];
            jobs_X.forEach(function(element){
                data_canvas.push({label: element[0], y: element[1], color: '#3C557F', indexLabelFontSize:16, indexLabelFontColor:'#3C557F'});
            });
            
            plotCanvas(data_canvas, "Private & Public Sectors", "Amount of jobs per category");
            
            var targetDiv = document.getElementById('sample-metadata');
            targetDiv.innerHTML = "<mark class='green'><b>PROS:</b></mark><ul><li><b>Flexibility:</b> More opportunities to change jobs in order to find what suits you best.</li><li><b>More Jobs:</b> Due to the number of businesses that make up the public/private sector, there are far more opportunities available.</li><li><b>Better Progression:</b> Easier to receive a pay rise and progress within a company.</li><li><b>Better Pay:</b> Salaries are more competitive and negotiable than in government.</li></ul><hr style='border:0.5px solid #a9a9a9'><mark class='red'><b>CONS:</b></mark><ul><li><b>Benefits:</b> Benefits depend on the generosity of your employer.</li><li><b>Big Competition:</b> A lot of competition from the time you look for a job and once you’re employed. Must keep in mind two things: 1) constantly thrive for being the best in your field, and 2) be aware of the likelihood of losing your job.</li><li><b>Possible Stressful Environment:</b> You’ll need to have the ability to deal with obstacles quickly and effectively.</li></ul>"
        });
    }
}

init();