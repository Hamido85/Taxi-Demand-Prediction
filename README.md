
# Taxi Demand Prediction

![AWS S3 Bucket](/Images/taxi.jpg)


* [1. Introduction](#1-Introduction)

* [2. Used data](#2-Used_data)

* [3. Used Tools](#3-Used_Tools)

* [4. Methodology](#4-Methodology)

* [4. Making forecasting app using streamlit framework](#5-Forecasting-streamlit-app)









# 1-Introduction

A taxi demand forecasting system was developed for Manhattan city that uses a combination of machine learning model and deep neural network model. It uses a Deep Neural Network model called **CNN-LSTM Encoder-Decoder** to forecast Manhattan's top 5 locations, while it utilizes the **Stacked Machine Learning** model to forecast the remaining locations.

* _It is worth noting, The models have been trained on historical taxi trips data from 01-01-2020 up until 01-04-2022_

App link: https://taxi-demand-prediction-363710.ew.r.appspot.com/
[![see it in action](/Images/app_picture_view.png)](https://youtu.be/fQa1YV3aN0U)


# 2-Used_data
## 2.1-Downloading the datasets
* The Taxi trips data used in this project was downloaded from [Taxi & Limousine Commission website]( https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) for the period between Jan-2020 to Apr-2022. 
* The weather data used in this project was downloaded from [visual crossing website](https://www.visualcrossing.com/) for the period between Jan-2020 to Apr-2022.
## 2.2-Understanding the datasets features
* The taxi dataset contains taxi trips records information for all the trips occurred in New York city such as `Pick-up time and date`,`Pick-up location`, `Drop-off location`, `Fare amount`, `Payment method` ..etc. [You can review the full features description from here](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
* The weather data contains hourly historical weather conditions such as `Temperature`, `Precipitation`,`Snow`,`Wind Speed`,`Humidity`. [You can review the full features description from here](https://www.visualcrossing.com/resources/documentation/weather-data/weather-data-documentation/)

# 3-Used_Tools
## 3.1 AWS EMR Cluster
* The taxi data contains more than 54 million records that can't be analyzed and processed using traditional methods. To get the most out of this data, we need to create __Amazon Elastic MapReduce Spark cluster__ to process the data and make it ready for modeling.
## 3.2 Tableau
* To analyze the transformed data and the weather data, we will be using `Tableau`.

## 3.3 Streamlit framework
* In order to build forecasting application, streamlit framework will be used.

## 3.4 Google Cloud platform
* To deploy the forecasting app and run it in the cloud.

# 4-Methodology
## 4.1 Data Preprocessing

* Create New S3 Bucket and upload the taxi data inside that Bucket folders
* Create the output folder which will contain the transformed data.

![AWS S3 Bucket](/Images/S3_Bucket.png)
![AWS S3 Bucket](/Images/taxi_data_folder.png)

* Create Amazon EMR cluster
The following Software configuration was chosen in this cluster

![AWS S3 Bucket](/Images/Spark_cluster.png)

Cluster Nodes and Instances: 
- 1 Master instance `m5.xlarge`
- 3 core instance `m5.xlarge`

* Open Jupyter notebook using `pyspark kernel` 
To view the preprocessing stage Spark Notebooks, open the following:

* (For 2020 Taxi data preprocessing open[Preprocess_2020](/Spark_preprocess_notebooks/Spark_preprocess_2020.ipynb))*.
* (For 2021 Taxi data preprocessing open[Preprocess_2021](/Spark_preprocess_notebooks/Spark_preprocess_2021.ipynb))*.
* (For 2022 Taxi data preprocessing open[Preprocess_2022](/Spark_preprocess_notebooks/Spark_preprocess_2022.ipynb))*.

## 4.2 Mering the aggregated taxi data with weathers and holidays
 
 * In this [notebook](/Merging_taxi_weather/Merging_and_EDA.ipynb), I have merged the Taxi data with weather data and holiday. I also performed some EDA analysis.
 
## 4.3 Exploratory data analysis using Tableau

 Now is time to explore the data further by creating some self-explanatory graphs.The goal is to understand how different weather conditions affect the pick-up trips.
 Also to analyze the pick-up trends over different periods of time to look for patterns.
 
 ### 4.3.1 Scatter Plots
 Below is a scatter plot to visualize the correlation between pick-up trips and temperature.
 from the chart, we can understand that taxi pick-up count increase when the temperature is moderate and it decreases when the temperature is very cold or very hot.

  ![Scatter_plot](/Images/Trips_count_vs_temp_scatter.png)

Below is a scatter plot to to visualize the correlation between pick-up trips and rain. 
from the chart, we can understand that taxi pick-up count increase when there is no rain and its start to decrease when there is rain. You can see how the pick-up trips drop dramatically when the rain precipitation is increasing

![Scatter_plot](/Images/pick_trips_vs_rain.png)

Below is a scatter plot to to visualize the correlation between pick-up trips and wind speed. 
from the chart, we can understand that taxi pick-up count decreases when the wind speed is increasing.

![Scatter_plot](/Images/pick_trips_vs_wind.png)


 ### 4.3.2 Pick-up trips trend line patterns over different periods of time

Average weekly pickups count trend line.
This graph shows that the average number of pickups decreases in Sunday and Monday and bit lower than the other days.

![trendline](/Images/weekly_average_trend_line.png)

The daily average pick-up count trend line shows the changes in the hours of the day. The peak hours are usually between 12 PM and 7 PM. Also, the lowest pick-up count is observed during the first few hours of the day.

![trendline](/Images/Hourly_average_pick_trend.png)

### 4.3.3 Top Manhattan zones in terms of taxi pick-up count

The below treemap shows the top zones in terms of taxi pick-up count.
We will develop Deep Neural Network model called CNN-LSTM Encoder-Decoder to forecast the pick-up trips in Manhattan's top 5 zones and develop Stacked Machine Learning model to forecast the pick-up trips in the remaining zones.

![treemap](/Images/pick-up_location_count.png)

## 4.4 Modeling part

The modeling part is divided into two sections, first section is to develop predictive model for 62 Manhattan's zones while the second section is to develop nueral network model for the top 5 zones as shown below

![Modeling](/Images/app_arch.png)

### 4.4.1 Machine learning modeling for 62 Manhattan's zones
In this [notebook](/Modeling//Final_ML_Modeling.ipynb), I analyze various predictive models to forecast the pick-up trips. Then choosed the best model based on evaulation metrics as shown below.
The used models are:
- Baseline model which is the average of pickups per zone and per hour.
- Linear Regression
- Decision Tree Regresor
- Random Forest Regresor
- XGboost regresor.
- Stacking Model. (This model combine several bagging and boosting models)

#### 4.4.1.1 Compare metrics
The following bar plot shows the best model.

![compare_models](/Images/compare_models.png)

As we can see, the stacking model outperform all the used model. 
below is the stacking model regression plot which shows how strong is the correlation between the prediction values and the true values
![compare_models](/Images/stacked_regression.png)

The below bar graph shows how close the predictions from the true values
![compare_models](/Images/prediction_vs_true.png)

### 4.4.2 Multivariate time series neural network model
In this [notebook](/Modeling/Final_DL_modeling.ipynb), I analyze various maltivariate neural models to forecast the pick-up trips in Manhattan's top 5 locations. Then choosed the best model based on evaulation metrics.
The used models are:
- Baseline model which is the average of pickups per zone and per hour.
- Encoder-Decoder Model
- CNN-LSTM Encoder Decoder Model

#### 4.4.2.1 Compare metrics

As we can see, CNN-LSTM-Encoder Decoder model outperformed the other models
below is the performance metrics comparison

 ![compare_models](/Images/DL_Models_metrics.png)

#### 4.4.2.2 CNN-LSTM Encoder Decoder prediction_vs_true
below is the CNN-LSTM Encoder Decoder prediction_vs_true for the top 5 locations

Location ID 161

 ![compare_models](/Images//location_161_prediction_vs_true.png)

 
Location ID 162

 ![compare_models](/Images//location_162_prediction_vs_true.png)

 Location ID 186

 ![compare_models](/Images//location_186_prediction_vs_true.png)

  Location ID 236

 ![compare_models](/Images//location_236_prediction_vs_true.png)

   Location ID 237

![compare_models](/Images//location_237_prediction_vs_true.png)

## 5-Forecasting-streamlit-app

## 5.1-Building forecasting app
Now after developing the necessary models, its time to develop forecasting app using the chosen models.
we are going to use streamlit framework to build the app and then we are going to deploy the app in google cloud.

in this [notebook](/streamlit_app/Streamlit_forecast_app.ipynb
) you will see the full streamlit app code. Also check the app folder which contain all the necessary files needed in this app.

## 5.2-Deploying the forecasting app in google cloud
Below is all the steps you will need to deploy the app in google cloud:

Create the app folder which will contain the following:

1- app.py ( the main python scripts –streamlit app)

2- Create requirements.txt file which contains all the libraries used in the app.

3- All the necessary files needed to run the app (the saved models, saved scaler,saved 24 samples, weather dataframe, and some other csv and geojson files which we used to create the map)

4- Create [Docker file](/streamlit_app/app_folder/Dockerfile). This is important to create image of the app with all the libraries and dependencies to run the it in the cloud.

5- Create [app.yaml file](/streamlit_app/app_folder/app.yaml). this is important to configure the used VM hardware which will host your app in the cloud.

6- Create Google cloud account.

7- Create New project inside google cloud.

8- Once all the files are ready inside the app folder, then open Google Cloud SDK Shell and navigate to your app folder path.

9- You need to set the Deploying directory path to the project you created in google cloud.
  - To know what is your current directory ==> gcloud config get-value project.
  
10- To set the project as the default directory for deployment
  - gcloud config set project {project name you wanna set} 
  
11- Now make sure that the default directory is the project you set in the above step
  - gcloud config get-value project ==> this should return the name of the project you have chosen.
  
12- Last step now is deploy the app
  - gcloud app deploy
  
13- You will be asked to choose region to deply the app, you might refer to this link to help you (https://googlecloudplatform.github.io/region-picker/)


Note: you might need to set the deployment runtime to 1200 since the defult runtime is just 10 mins. sometimes if your deployment will take more than 10 mins it will fail to deploy.
  - cloud_build_timeout 1200

Please note that I added second page in streamlit app which contains information about the models and the evaluation metrics, the full code is available inside app_folder [app.py](/streamlit_app/app_folder/app.py).
(check the app folder which contain all the files)[/streamlit_app/app_folder/].



**Click in the image to see it in action!**  

[![see it in action](/Images/app_picture_view.png)](https://youtu.be/fQa1YV3aN0U)


## Authors

- [@Hamido85](https://www.github.com/Hamido85)
