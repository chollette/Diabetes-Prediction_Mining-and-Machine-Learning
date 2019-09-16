# Diabetes-Prediction_Mining-and-Machine-Learning

### The On-Set of Diabetes Prediction using PIMA INDIANs Dataset
There has been series of predictions for the PIMA INDIAN Diabetes dataset, but none has been able to create a process that generally works for predicting diabetes in the real sense.

I present an approach that is based on the following medical facts about diabetes diagnosis.

###  What tests are used to diagnose diabetes and prediabetes?
Health care professionals most often use the fasting plasma glucose (FPG) test or the A1C test to diagnose diabetes. In some cases, they may use a random plasma glucose (RPG) test.

#### Fasting plasma glucose (FPG) test
The FPG blood test measures your blood glucose level at a single point in time. For the most reliable results, it is best to have this test in the morning, after you fast for at least 8 hours. Fasting means having nothing to eat or drink except sips of water.

##### Glucose level for diabetes diagnosis: 
<99 is Normal; 100-125 is prediabetes; >= 126 & <70 is Diabetes
[Source: https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis#type1]

### What test numbers indicate the type of diabetes? 
Insulin is a hormone that is responsible for allowing glucose in the blood to enter cells, providing them with the energy to function. The test values are as given below:
          Normal Type 1 Type 2
          30-320 20-25 90-230
Type 1 diabetes - Insulin is expected to be lower, as this relates to the destruction of the insulin-producing cells. Insulin is insufficient or non-existent in the body, and a person with type 1 diabetes will need to take insulin on a life-long basis. 
Type 2 diabetes The amount of insulin in your blood is higher than what's considered normal. 
[Sources: https://www.medicalnewstoday.com/articles/323760.php; https://www.endocrineweb.com/conditions/type-1-diabetes/what-insulin]

For the above reasons, only the glucose and insulin will be used to predict diabtes and type of diabetes.

This approach so far achieves the best predition accuracy on PIMA Indian Diabetes Dataset, it is provided below. If a new instance is diabetic, can classify the type of diabetes the new instance have.
##### Train acc:  0.9508196721311475
##### Test acc:  0.9607843137254902


## To run the codes, follow the below given sequence of instruction:
#### (1) Download the PIMA INDIAN Diabetes Dataset (https://www.kaggle.com/uciml/pima-indians-diabetes-database) or load provided data
#### (2) Data Analysis, Visualization and Interpretation
#### (3) Create Regression Data
#### (4) Predict Missing Values Using Linear Regression
#### (5) Diabetes Prediction
#### (6) Diabetes Type Classifier
