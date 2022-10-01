
# Taxi Demand Prediction

![AWS S3 Bucket](/Images/taxi.jpg)


* [1. Introduction](#1-Introduction)

* [2. Used data](#2-Used_data)

* [3. Used Tools](#3-Used_Tools)

* [4. Methodology](#4-Methodology)









# 1-Introduction

A taxi demand forecasting system was developed for Manhattan city that uses a combination of machine learning model and deep neural network model. It uses a Deep Neural Network model called **CNN-LSTM Encoder-Decoder** to forecast Manhattan's top 5 locations, while it utilizes the **Stacked Machine Learning** model to forecast the remaining locations.

* _It is worth noting, The models have been trained on historical taxi trips data from 01-01-2020 up until 01-04-2022_

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
*([Preprocess_2020](https://github.com/Hamido85/Taxi-Demand-Prediction/Spark_preprocess_notebooks/Spark_preprocess_2020.ipynb))*.
