from turtle import width
from keras.models import load_model
import pickle
import tensorflow as tf
import holidays
import pandas as pd
import numpy as np
from pickle import load
import streamlit as st
import folium
from streamlit_folium import st_folium
import altair as alt
from streamlit_option_menu import option_menu
from PIL import Image
import plotly
import plotly.express as px
import plotly.graph_objects as go


APP_TITLE = 'Manhattan Taxi Demand Prediction'
APP_SUBTITLE = "A taxi demand forecasting system was developed for Manhattan city that uses a combination of machine learning model and deep neural network model. It uses a Deep Neural Network model called CNN-LSTM Encoder-Decoder to forecast Manhattan's top 5 locations, while it utilizes the Stacked Machine Learning model to forecast the remaining locations."

#================================ Define Functions for the APP ===================================================================

# 1- This function will be used to add locationID coloum to the weather data for all the locations used in the machine learning model

def locationsID_Generate_ml_model(dataframe,locationID):
    df = dataframe.copy(deep=True)
    df['LocationID']=locationID
    df.reset_index(inplace = True,drop=True)
    return df

# 2- This function will be used to transform the the deep learning model prediction to the same format of the machine learning model predction so we will be able to concate them by rows.
def locationsID_Generate_deep_model(dataframe,location):
    df = dataframe.copy(deep=True)
    df['LocationID']=location
    df['Trips_Forcast'] = df[location]
    df = df[['LocationID','datetime','Trips_Forcast']]
    df.reset_index(inplace = True,drop=True)
    return df

# 3- Prepare the weather data for the ML model
@st.cache
def prepare_weather_data(dataframe):
    '''
    The ML model expect the data to contain the following coloumns:
    (LocationID,Year, Month, DayOfMonth, Hour, dayofweek, temp, humidity, precip, snow, windspeed, Holiday, IsWeekend)
    so we will preprare the forecast data to have this formate
    '''

    
    dataframe['datetime']= dataframe['datetime'].astype('datetime64[ns]')
    dataframe['datetime1'] = dataframe['datetime'].dt.strftime('%Y-%m-%d%-H%M')
    # filtering the weather data from 1-May-2022 onward
    dataframe=dataframe[(dataframe['datetime'] >= '2022-05-01')]
    
    # Create features from datetime column
    dataframe['Year'] = dataframe['datetime'].dt.year
    dataframe['Month'] = dataframe['datetime'].dt.month
    dataframe['DayOfMonth'] = dataframe['datetime'].dt.day
    dataframe['Hour'] = dataframe['datetime'].dt.hour
    dataframe['dayofweek'] = dataframe['datetime'].dt.dayofweek
    
    # create IsWeekend feature
    dataframe["IsWeekend"] = dataframe["dayofweek"] >= 5
    dataframe['IsWeekend'].replace({True:1,False:0}, inplace=True)
    
    # create date string column ( we will use this feature to merge this data with holiday data)
    dataframe['date']=dataframe['datetime'].apply(lambda x: x.strftime('%d%m%Y'))
    
    # create holiday column, creating holidays dataset based on United states using holiday package
    
    
    holiday_list = []
    for holiday in holidays.UnitedStates(years=[2022]).items():
        holiday_list.append(holiday)
    
    # this contain all the holiday for 2022
    holidays_df = pd.DataFrame(holiday_list, columns=["date", "holiday"])
    # creating int holiday coloum not string
    holidays_df['holiday']=1
    
    # now creating the same date string column to merge with the weather data
    holidays_df['date']=holidays_df['date'].apply(lambda x: x.strftime('%d%m%Y'))
    
    # join holiday with weather data
    dataframe=dataframe.merge(holidays_df, on='date', how='left')
    # filling the nan values in holiday with zero (zero means no holiday, 1 means there is holiday)
    dataframe['holiday'] = dataframe['holiday'].fillna(0)
    
    # now we will create data set which will have the merged data plus location ID column 
    
    location_list = [263, 262, 261, 249, 246, 244, 243, 239, 238, 234, 233, 232, 231, 230, 229, 224, 211, 209, 202, 194, 170, 166, 164, 163, 158,
    153, 152, 151, 148, 144, 143, 142, 141, 140, 137, 128, 127, 125, 120, 116, 114, 113, 107, 105, 100, 90, 88, 87, 79, 75, 74,
    68, 50, 48, 45, 43, 42, 41, 24, 13, 12, 4]
    
    # create empty dataframe
    df=pd.DataFrame()
    for i in location_list:
        generate_location=locationsID_Generate_ml_model(dataframe,i)
        df = pd.concat([df, generate_location], axis=0)
        #df = df.concat(generate_location)
    
    # now choosing the same features the ML model was trained on
    df_forcast = df[['LocationID','datetime','Year', 'Month', 'DayOfMonth', 'Hour', 'dayofweek',
       'temp', 'humidity', 'precip', 'snow', 'windspeed', 'holiday',
       'IsWeekend']]
    return df_forcast

# 4- forecast function using ML and DL models and concating the prediction of the two models
@st.cache
def data_forcast(dataframe,days_to_forcast):

# loading the ML and DL models

    with open('ML_model.pkl' , 'rb') as f:
        ml_model = pickle.load(f)
    
    dl_model = load_model("CNN_Encoder_Decoder_final_model.h5")
    
                    #================= Forcast using the Machine Learning Model===============================
    
    # here we need to filter out the prepared data for machine learning to match with the forcast time range
    # since the ML model was trained for the weather data from January 2020 till April-2022, then we need to filter out the
    # prepared data from 1-may-2022 till the end of the forecasted dates.
    
    filtered_data = dataframe.loc[(dataframe['Month']==5) & (dataframe['DayOfMonth']>= 1) & (dataframe['DayOfMonth']< days_to_forcast+1)]
    preparded_data=filtered_data.drop(['datetime'], axis=1)
    
    # make prediction using ML model
    ML_forcast = ml_model.predict(preparded_data)
    # convert the prediction to integer since we have float data type and add the prediction to the filtered data
    filtered_data['Trips_Forcast']=ML_forcast.astype(int)
    
    # choose the the wanted coloumns 'LocationID', 'datetime', 'Trips_Forcast'
    filtered_data=filtered_data[['LocationID', 'datetime', 'Trips_Forcast']]
    
    # since we have negative prediction, we will convert any negative value to zero
    filtered_data['Trips_Forcast'].mask(filtered_data['Trips_Forcast'] < 0, 0, inplace=True)
    
    
                    #================= Forcast using the Deep Learning Model===============================
    
    # since the deep learning model forecast hourly data, then the forecast range will be 24* days to forecast
    FORCAST = 24*days_to_forcast
    
    # to be able to predict the values from 1-May forward, we will need the last 24 hours from the normalize trainin data
    X_test_mod = np.load('last_24.npy')
    X_test_mod = X_test_mod.reshape((1, 24, 5))
    y_preds = []
    for n in range(FORCAST):
        y_pred = dl_model.predict(X_test_mod, verbose=0)
    
        X_test_mod = np.append(X_test_mod, y_pred, axis=1)
        X_test_mod = X_test_mod[:,1:, :]
        y_preds = np.append(y_preds, y_pred)
    
    y_preds_reshaped = y_preds.reshape(-1,5)
    
    # we will inverse the normalize values using saved scaler.
    scaler = load(open('scaler1.pkl', 'rb'))
    y_preds_inverse = scaler.inverse_transform(y_preds_reshaped)
    
     # convert the prediction to integer since we have float data type
    y_preds_inverse = y_preds_inverse.astype(int)
    
    # create dataframe from the prediction
    forcast_df = pd.DataFrame(data=y_preds_inverse,columns=[161,162,186,236,237])
    
    # create timeseries series from 1-May-2022 till the end of forecast range
    future_date=pd.date_range('2022-05-01', periods=FORCAST, freq='H')
    
    # adding the timeseries to the forecast dataframe
    forcast_df['datetime'] = future_date
    
    # now we will transform the forecast dataframe to the same output format of the ML prediction to concate them together
    locations=[i for i in forcast_df.columns if i != 'datetime'] # to choose only the 5 locations 
    forcast_df_transfrom=pd.DataFrame()
    for i in locations:
        generate_location=locationsID_Generate_deep_model(forcast_df,i)
        
        forcast_df_transfrom = pd.concat([forcast_df_transfrom, generate_location], axis=0)
        #forcast_df_transfrom = forcast_df_transfrom.append(generate_location)
        
    # since we have negative prediction, we will convert any negative value to zero    
    forcast_df_transfrom['Trips_Forcast'].mask(forcast_df_transfrom['Trips_Forcast'] < 0, 0, inplace=True)
    
    # now concating the ML prediction with DL prediction
    combined_forcast = pd.concat([filtered_data, forcast_df_transfrom], axis=0)
    
    # create Month, hour features
    combined_forcast['DayOfMonth'] = combined_forcast['datetime'].dt.day
    combined_forcast['Hour'] = combined_forcast['datetime'].dt.hour
    
    # to be able to visualize the hourly prediction in the map, I have noticed that locationID 105 is not available in the Geojson file
    # which will cause errors later when we use folium map. so I had to remove the prediction for this location
    combined_forcast = combined_forcast[~combined_forcast['LocationID'].isin([105])]
    
    # now converting the LocationID to string
    combined_forcast['LocationID'] = combined_forcast['LocationID'].apply(str)

    return combined_forcast


#=================== Creating the folium map to display the hourly prediction ==================

def display_map(dataframe,day,hour):
    dataframe = dataframe.loc[(dataframe['DayOfMonth']==day) & (dataframe['Hour']== hour)]
    dataframe=dataframe.drop(['datetime'], axis=1)

    map = folium.Map(location=[40.7831, -73.9712], zoom_start=11.4,scrollWheelZoom=False,tiles='CartoDB positron')
    choropleth = folium.Choropleth(
        geo_data='manhattan_zones.geojson',
        data=dataframe,
        columns=('LocationID','Trips_Forcast'),
        key_on='feature.properties.location_id',
        fill_color="YlOrRd",
        line_opacity=1,
        highlight=True
        
        )
    choropleth.geojson.add_to(map)

    dataframe=dataframe.set_index('LocationID')


    for feature in choropleth.geojson.data['features']:
        Location_ID = feature['properties']['location_id']
        feature['properties']['Trips_Forcast'] =  str(dataframe.loc[Location_ID,'Trips_Forcast'])
    


    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['location_id','zone','Trips_Forcast'],
            aliases=['Location ID : ', 'Area Name : ', 'Forcasted Taxi trips : '])
    )
    st_map = st_folium(map,width = 700, height = 450)

#======================== Create time series trend line chart using altair ======================

def line_chart(df):
    brush = alt.selection(type='interval', encodings=['x'])
    base = alt.Chart(df).mark_line().encode(
    x = 'datetime:T',
    y = 'Trips_Forcast:Q'
    ).properties(
    width=600,
    height=200)
    upper = base.encode(
    alt.X('datetime:T', scale=alt.Scale(domain=brush)))
    lower = base.properties(
    height=60).add_selection(brush)

    return alt.vconcat(upper, lower)

# ====================== Create bar chart to display the top location ===========================

def bar_plot(dataframe):

    bar =alt.Chart(dataframe).mark_bar().encode(
    x='Trips_Forcast:Q',
    y=alt.Y('Zone:N', sort='-x'),
    color=('Trips_Forcast:Q'),
    tooltip=['Trips_Forcast'])
    return bar

def Hourly_Growth(df,day,hour):
    
    if hour==0:
        
        growth=str(0)
        growth=growth+' '+'%'
        
        return growth
    else:
        end_value = df[(df['DayOfMonth']==day) & (df['Hour']==hour)]['Trips_Forcast'].sum()
        start_value = df[(df['DayOfMonth']==day) & (df['Hour']==hour-1)]['Trips_Forcast'].sum()
        growth = str(round((((end_value - start_value)/start_value)*100),1))
        growth=growth+' '+'%'
        
        return growth

def Daily_Growth(df,day):
    if day==1:
        growth=str(0)
        return growth
    else:
        end_value = df[(df['DayOfMonth']==day)]['Trips_Forcast'].sum()
        start_value = df[(df['DayOfMonth']==day-1)]['Trips_Forcast'].sum()
        growth = str(round((((end_value - start_value)/start_value)*100),2))
        growth=growth+' '+'%'
        return growth

#==================================Finish creating functions for the App=============================================================

#======================================== The main streamlit app  ================================

def main():
    st.set_page_config(APP_TITLE)
    st.text('Author: Abdelhamid Shaat')
    #st.title(APP_TITLE)
    #st.caption(APP_SUBTITLE)
    #st.caption(f'It is worth noting, The models have been trained on historical taxi trips data from 01-01-2020 up until 01-04-2022' )

    with st.sidebar:
        selected = option_menu(
            menu_title='Main Menu',
            options=['Demand Prediction App', "More information about the App"],
            icons=['tv','book'],
            #default_index=0,
        )
    
    if selected == 'Demand Prediction App':
        st.title(APP_TITLE)
        st.caption(APP_SUBTITLE)
        st.caption(f'It is worth noting, The models have been trained on historical taxi trips data from 01-01-2020 up until 01-04-2022' )


    # reading weather data
        df_temp=pd.read_csv('new york_weather.csv',usecols=['datetime','temp','humidity','precip','snow','windspeed'])

    
    # process the weather data
        df_forcast = prepare_weather_data(df_temp)


    # create sidebar select for how many days to forecast in the future

        Forecast_list = [3,7,14]

        st.sidebar.caption('Forecast starts from 01-May-2022')

        Forecast_days = st.sidebar.selectbox('No. of days to be forecasted',Forecast_list)

    # this will do the prediction, just by spacifiying how many days to forecast 3,7,14

        combined_forcast = data_forcast(df_forcast,Forecast_days)
    

    # creating the forecasted days in the list to create daily sidebar selector
        day_list = list(combined_forcast['DayOfMonth'].unique())
        Filter_days = st.sidebar.selectbox('Select the day to view on map',day_list)

    # creating the forecasted hours 0,23 in the list to create hourly sidebar selector
        hour_list = list(combined_forcast['Hour'].unique())
        Filter_hour = st.sidebar.selectbox('Select the forecasted hour to view on map',hour_list)

        st.subheader(f'Selected date:   {Filter_days}-May-2022: {Filter_hour}:00' )

    # creating metrics
        Daily_GR = Daily_Growth(combined_forcast,Filter_days)

        Hourly_GR = Hourly_Growth(combined_forcast,Filter_days,Filter_hour)
    

        col1,col2,col3 = st.columns(3)
        with col1:
            Total_daily_trips = combined_forcast[(combined_forcast['DayOfMonth']==Filter_days)]['Trips_Forcast'].sum()
            st.metric("No. of predicted trips during this day",'{:,}'.format(Total_daily_trips),Daily_GR)
        with col2:
            Total_hourly_trips = combined_forcast[(combined_forcast['DayOfMonth']==Filter_days) & (combined_forcast['Hour']==Filter_hour)]['Trips_Forcast'].sum()
            st.metric("No. of predicted trips during this hour",'{:,}'.format(Total_hourly_trips),Hourly_GR)
        with col3:
            percentage = str(round(((Total_hourly_trips/Total_daily_trips)*100),2)) +' '+'%'
            st.metric(" Percentage of trips during this hour",percentage)
    

    # displaying the map
        st.subheader(f'Manhattan Map' )
        display_map(combined_forcast,Filter_days,Filter_hour)


    # displaying top 10 hourly location forcasted count
        st.subheader(f'Hourly Top 10 locations')
        st.caption(f'Top 10 locations according to the number of forcasted trips during the selected hour {Filter_hour}:00' )
    


        Hourly_top_counts = combined_forcast[(combined_forcast['DayOfMonth']==Filter_days) & (combined_forcast['Hour']==Filter_hour)]
        Hourly_top_counts=Hourly_top_counts.sort_values("Trips_Forcast", axis = 0, ascending = False, na_position ='first')
    # reading location lookup file
        location_lookup=pd.read_csv('taxi_Zone_lookup.csv',usecols=['LocationID','Zone'])
        location_lookup['LocationID'] = location_lookup['LocationID'].astype(str)
    # merging the data
        Hourly_top_counts=Hourly_top_counts.merge(location_lookup, on='LocationID', how='left')
    # chooing top 10 N-largest values
        Hourly_top_counts = Hourly_top_counts.nlargest(10, "Trips_Forcast")
        Hourly_top_bar_chart=bar_plot(Hourly_top_counts)
        st.altair_chart(Hourly_top_bar_chart, use_container_width=True)


        st.header(f'Daily analysis' )
        st.subheader(f'Line chart trend for the forecasted period' )
    
    # displating the line chart
        group=combined_forcast.groupby('datetime')[['Trips_Forcast']].sum()
        group.reset_index(inplace=True)
        chart_data = line_chart(group)
        st.altair_chart(chart_data, use_container_width=True)

    # displaying daily top 10 location by bar chart
        st.subheader(f'Daily Top 10 locations')
        st.caption(f'Top 10 locations according to the number of forcasted trips on {Filter_days}-May-2022' )

        Daily_top_counts = combined_forcast[(combined_forcast['DayOfMonth']==Filter_days)]
        Daily_top_counts=Daily_top_counts.groupby('LocationID')[['Trips_Forcast']].sum()
        Daily_top_counts.reset_index(inplace=True)
        Daily_top_counts = Daily_top_counts.sort_values("Trips_Forcast", axis = 0, ascending = False, na_position ='first')
        Daily_top_counts['LocationID'] = Daily_top_counts['LocationID'].astype(str)
        Daily_top_counts=Daily_top_counts.merge(location_lookup, on='LocationID', how='left')
        Daily_top_counts = Daily_top_counts.nlargest(10, "Trips_Forcast")
        Daily_top_bar_chart=bar_plot(Daily_top_counts)
        st.altair_chart(Daily_top_bar_chart, use_container_width=True)

        #======================= SECOND PAGE IN THE APP=====================================

    if selected == 'More information about the App':

        st.header('How does it work')
        st.subheader('Taxi demand forecast system architecture')
        st.caption('The Taxi demand forecast system utilizes two trained models to forecast the number of taxi trips in the future. One of them is the stacked machine learning model and the other is deep neural network model called the CNN-LSTM Encoder-Decoder which it’s considered one of the best sequence learning models. Forecasting the top 5 locations in terms of the number of trips occurs through the CNN-Encoder-Decoder model; while, forecasting the trips in the remaining locations occurs through the stacked machine learning model as shown below. ')
        img = Image.open('app_arch.png')
        st.image(img,width=800,caption='App architecture')

        st.subheader('1- The stacked Model architecture ')
        st.caption('The stacked model consists of a combination of boosting and bagging 3 base learner models, namely Decision tree regressor, Random Forest regressor, and Xgboost regressor as shown below. ')
        img = Image.open('stack_arch.png')
        st.image(img,width=800,caption='Stacked Model')

        st.markdown('#### Stacked model evaluation metrics')
        st.caption('When the stacked model performance was evaluated against a couple of standalone models on a four-months taxi trips data, it outperformed other standalone models.  The evaluation metrics used are the R-Square, Mean Square Error, Mean Absolute Error, and Root Mean Square Error.')
        
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.metric("R-Square",'96.72 %')
        with col2:
            st.metric("Mean Square Error",'125.51')
        with col3:
            st.metric("Mean Absolute Error",'5.82')
        with col4:
            st.metric("Root Mean Square Error",'11.2')

    
        ML_metrics= pd.read_csv('ML_metrics.csv')
        
        st.markdown(' R-Square comparison between Stacked model and other standalone models ')
        bar_R2 =alt.Chart(ML_metrics).mark_bar().encode(
        x='R-Square:Q',
        y=alt.Y('Models:N', sort='-x'),
        color=('R-Square:Q'),
        tooltip=['R-Square'])
        st.altair_chart(bar_R2, use_container_width=True)

        st.markdown(' RMSE comparison between Stacked model and other standalone models ')
        bar_RMSE =alt.Chart(ML_metrics).mark_bar().encode(
        x='RMSE:Q',
        y=alt.Y('Models:N', sort='x'),
        color=('RMSE:Q'),
        tooltip=['RMSE'])
        st.altair_chart(bar_RMSE, use_container_width=True)

        st.markdown('#### Stacked model Predicted trips value Vs the true trips value')
        img = Image.open('stacked prediction.png')
        st.image(img,width=800,caption='Prediction Vs True')

        st.markdown('Regression plot')
        st.caption('A regression plot is a statistical method that shows a relationship between the predicted value and the true value.')
        img = Image.open('stack regression plot.png')
        st.image(img,width=500,caption='Stacked Model Regression Plot')


        st.subheader('2- The CNN LSTM Encoder-Decoder Model architecture ')
        st.caption('The CNN encoder-decoder is an extension of the existing encoder-decoder architecture which is used for sequence learning. In our case, it’s a multivariate time series for the top 5 locations. This model consists of two parts Encoder and Decoder. The encoder is mainly responsible for interpreting and reading the input. It consists of two Conv1D layers to extract features, and then it is flattened after pooling. The RepeatVector layer is used to repeat the context vector resulted from the encoder part. The output of the decoder is then connected to a fully connected Dense layer via the TimeDistributed layer which separates the output for each time step. ')
        img = Image.open('cnn_enc_dec-arch.png')
        st.image(img,width=500,caption='CNN Encoder-Decoder Model architecture')

        st.markdown('#### CNN LSTM Encoder-Decoder model learning curve ')
        st.caption('Learning curves are plots that show changes in learning performance over time in terms of experience. it show the CNN Encoder-Decoder model performance on the train and validation datasets ')
        img = Image.open('cnn_enc_dec-loss.png')
        st.image(img,width=500,caption='CNN-Encoder-Decoder model learning curve')

        st.markdown('#### CNN LSTM Encoder-Decoder model evaluation metrics')
        st.caption('When the CNN-Encoder-Decoder model performance was evaluated against the encoder-decoder model and baseline model on a four-months taxi trips data, it outperformed the other models.  ')

        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.metric("R-Square",'95.6 %')
        with col2:
            st.metric("Mean Square Error",'839.1')
        with col3:
            st.metric("Mean Absolute Error",'19.56')
        with col4:
            st.metric("Root Mean Square Error",'28.96')

        

        DL_metrics= pd.read_csv('DL_metrics.csv')
        st.markdown(' R-Square comparison between CNN Encoder Decoder model and other models ')

        bar_R2 =alt.Chart(DL_metrics).mark_bar().encode(
        x='R-Square:Q',
        y=alt.Y('Models:N', sort='-x'),
        color=('R-Square:Q'),
        tooltip=['R-Square'])
        st.altair_chart(bar_R2, use_container_width=True)

        st.markdown(' RMSE comparison between CNN LSTM Encoder Decoder model and other models ')
        bar_RMSE =alt.Chart(DL_metrics).mark_bar().encode(
        x='RMSE:Q',
        y=alt.Y('Models:N', sort='x'),
        color=('RMSE:Q'),
        tooltip=['RMSE'])
        st.altair_chart(bar_RMSE, use_container_width=True)


        st.markdown('#### CNN LSTM Encoder Decoder Predicted trips value Vs the true trips ')
        st.caption('The below line chart visualizes the CNN LSTM Encoder-Decoder model prediction trips count  VS the True taxi trips count when testing on a four-months taxi trips data in the top 5 locations in Manhattan. Each location is annotated by its location ID on the side select option in the chart. ')

        ytrue_predicted = pd.read_csv('cnn_enc_dec_prediction.csv')

        fig = go.Figure()

        columns_pred_vs_true=[i for i in ytrue_predicted.columns if i != 'timestamp']

        for column in columns_pred_vs_true:
            fig.add_trace(
            go.Scatter(
            x = ytrue_predicted['timestamp'],
            y = ytrue_predicted[column],
            name = column))

        fig.update_layout(
            updatemenus=[go.layout.Updatemenu(
                active=0,
                buttons=list(
                    [dict(label = '161',
                        method = 'update',
                        args = [{'visible': [True, False, False, False,False,True, False, False, False,False]},
                          {'title': '161 true Vs Predicted',
                           'showlegend':True}]),
                    dict(label = '162',
                        method = 'update',
                        args = [{'visible': [False, True, False, False,False,False, True, False, False,False]}, 
                          {'title': '162 true Vs Predicted',
                           'showlegend':True}]),
                    dict(label = '186',
                        method = 'update',
                        args = [{'visible': [False,False, True, False, False,False,False, True, False, False]},
                          {'title': '186 true Vs Predicted',
                           'showlegend':True}]),
                    dict(label = '236',
                        method = 'update',
                        args = [{'visible': [False, False, False, True,False,False, False, False, True,False]},
                          {'title': '236 true Vs Predicted',
                           'showlegend':True}]),
                    dict(label = '237',
                        method = 'update',
                        args = [{'visible': [False, False, False,False, True,False, False, False,False, True]},
                          {'title': '237 true Vs Predicted',
                           'showlegend':True}]),
                    ])
            )

            ])
        st.plotly_chart(fig,width=800)  

#======================================================================================================================================



    














    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ =="__main__":
    main()
