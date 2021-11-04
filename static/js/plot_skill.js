let skills=["Scheduling",'Communication','Customer Service','Sales',"Management ","Training","Hardworking And Dedicated",
            "Complex Problem Solving ","Operations","Attention To Detail","Self Motivation","Coordinating",
            "Leadership ","Maintenance ","Merchandising ","Retailing","Health Care",
            "Reports","Finance","Testing ", "Recruitment","Team-working ","Nursing","Information Security","Microsoft Excel ",
            "English Language"," Verbal Communication", 
            " Trustworthy","Administration","Registered Nurse Certified","Accounting"
]

let count=[2503,2070,1949,1670,1560,1555,1202,1062,881,876,842,840,839,829,814,765,737,712,679,
            673,667,664,619,612,607,606,595,575,559,538]

            

            function plot_skill(x,y){
                let trace1 = {
                  x: x,
                  y: y,
                  type: "bar"
                }
            
                let data=[trace1]
            
                let layout={
                    title:"Most In-Demand Skills"
                };
            
                Plotly.newPlot("plot1",data,layout);
            }
            plot_skill(skills,count)