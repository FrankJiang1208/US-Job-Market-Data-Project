d3.json("../static/resources/Salary_Prediction.json").then(function(data) {
    const job=data.Job_Title;
    const loc=data.State;
    const sec=data.Sector;
    const ind=data.Industry;
    const own=data.Ownership;
    const sen=data.Seniority;
    dropdown1=d3.select("#selDataset1");
    dropdown2=d3.select("#selDataset2");
    dropdown3=d3.select("#selDataset3");
    dropdown4=d3.select("#selDataset4");
    dropdown5=d3.select("#selDataset5");
    dropdown6=d3.select("#selDataset6");
    for (i=0;i<job.length;i++) {
        let op=dropdown1.append("option").text(job[i])
        op.property('value',job[i])
    }
    for (i=0;i<loc.length;i++) {
        let op1=dropdown2.append("option").text(loc[i])
        op1.property('value',loc[i])
    }
    for (i=0;i<sec.length;i++) {
        let op2=dropdown3.append("option").text(sec[i])
        op2.property('value',sec[i])
    }
    for (i=0;i<ind.length;i++) {
        let op3=dropdown4.append("option").text(ind[i])
        op3.property('value',ind[i])
    }
    for (i=0;i<own.length;i++) {
        let op4=dropdown5.append("option").text(own[i])
        op4.property('value',own[i])
    }
    for (i=0;i<sen.length;i++) {
        let op5=dropdown6.append("option").text(sen[i])
        op5.property('value',sen[i])
    }


    // d3.select("#submit").on("click", getInfo);

    // function getInfo(){
    //     let title=d3.select("#selDataset1").node().value

    //     let state=d3.select("#selDataset2").node().value

    //     let industry=d3.select("#selDataset3").node().value

    //     let size=d3.select("#selDataset4").on("change", change)

    //     let age=d3.select("#selDataset5").on("change", change)


    //     function change() {
    //         return this.value

    //     }

    //     if(d3.select("#Python").property("checked")){
    //         python = 1
    //     } else {
    //         python = 0	
    //     }	

    //     if(d3.select("#R").property("checked")){
    //         r = 1
    //     } else {
    //         r = 0	
    //     }	

    //     if(d3.select("#SQL").property("checked")){
    //         sql=1
    //     } else {
    //         sql=0		
    //     }	
            
    //     if(d3.select("#AWS").property("checked")){
    //         aws=1
    //     } else {
    //         aws=0		
    //     }	
            
    //     if(d3.select("#Excel").property("checked")){
    //         excel=1
    //     } else {
    //         excel=0	
    //     }	
            
    //     if(d3.select("#GCP").property("checked")){
    //         gcp=1
    //     } else {
    //         gcp=0		
    //     }	
            
    //     if(d3.select("#Azure").property("checked")){
    //         azure=1
    //     } else {
    //         azure=0		
    //     }	
            
    //     if(d3.select("#Spark").property("checked")){
    //         spark=1
    //     } else {
    //         spark=0		
    //     }	
            
    //     if(d3.select("#PyTorch").property("checked")){
    //         pytorch=1
    //     } else {
    //         pytorch=0		
    //     }	
            
    //     if(d3.select("#NoSQL").property("checked")){
    //         nosql=1
    //     } else {
    //         nosql=0		
    //     }	
            
    //     if(d3.select("#TensorFlow").property("checked")){
    //         tensorflow=1
    //     } else {
    //         tensorflow=0		
    //     }
        
    //     if(d3.select("#Tableau").property("checked")){
    //         tableau=1
    //     } else {
    //         tableau=0		
    //     }	

    //     if(d3.select("#Scikit_Learn").property("checked")){
    //         scikit_learn=1
    //     } else {
    //         scikit_learn=0		
    //     }	

    //     if(d3.select("#ml").property("checked")){
    //         machine_learning=1
    //     } else {
    //         machine_learning=0		
    //     }	

    //     if(d3.select("#Hadoop").property("checked")){
    //         hadoop=1
    //     } else {
    //         hadoop=0		
    //     }	

    //     if(d3.select("#Scala").property("checked")){
    //         scala=1
    //     } else {
    //         scala=0		
    //     }	

    //     if(d3.select("#DataBrick").property("checked")){
    //         databrick=1
    //     } else {
    //         databrick=0		
    //     }

        

    //     let jsonData={"Title":title,"State":state,"Industry":industry,"Age":age,"Size":size,
    //     "Python":python,"R":r,"SQL":sql,"AWS":aws,"Excel":excel,
    //     "GCP":gcp,"Azure":azure,"Spark":spark,"PyTorch":pytorch,
    //     "NoSQL":nosql,"TensorFlow":tensorflow,"Tableau":tableau,
    //     "Scikit Learn":scikit_learn,"Machine Learning":machine_learning,
    //     "Hadoop":hadoop,"Scala":scala,"Data Brick":databrick}

    //     $.ajax({
    //         type: "POST",
    //         contentType: "application/json;charset=utf-8",
    //         url: "/salary_prediction",
    //         traditional: "true",
    //         data: JSON.stringify({jsonData}),
    //         dataType: "json"
    //         });

    
    // }


    

});