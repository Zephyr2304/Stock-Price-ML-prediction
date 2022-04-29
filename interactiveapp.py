import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
from keras.models import load_model
from datetime import date
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objects as go

st.title('stock trend prediction')
User_input = st.sidebar.text_input("Enter stock Ticker", 'SBIN.NS')
start = st.sidebar.date_input("enter starting date ", date(2011, 9, 27))
end = date.today()

df = web.DataReader(User_input, 'yahoo', start, end)

# describing data

# print(df.reset_index()['Date'])


st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

st.subheader('Today price')
st.write(df.tail(1))

# Visualization

# stock data
st.subheader("Raw stock data")
raw = go.Figure()
raw.add_trace(go.Scatter(x=df.reset_index()['Date'], y=df['Open'], name=f'{User_input} Opening price'))
raw.add_trace(go.Scatter(x=df.reset_index()['Date'], y=df['Close'], name=f'{User_input} Closing price'))
raw.add_trace(go.Scatter(x=df.reset_index()['Date'], y=df['High'], name=f'{User_input} High price'))
raw.add_trace(go.Scatter(x=df.reset_index()['Date'], y=df['Low'], name=f'{User_input} Low price'))
raw.layout.update(title_text=f"{User_input} Stock Data", xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(raw)

# Closing price chart
st.subheader('Closing Price vs Time chart')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.reset_index()['Date'], y=df['Close'], name='Closing Price'))

fig.layout.update(title_text=f"{User_input} Closing Price", xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(fig)


st.subheader('Closing Price vs Time chart with 100MA')
fig2 = go.Figure()
ma100 = df.Close.rolling(100).mean()
fig2.add_trace(go.Scatter(x=df.reset_index()['Date'], y=df['Close'], name='stock_close'))
fig2.add_trace(go.Scatter(x=df.reset_index()['Date'], y=ma100, name='100 days MA'))
fig2.layout.update(title_text=f"{User_input} Close VS 100days MA", xaxis_rangeslider_visible=True, width=1000, height=500)
plt.plot(ma100, 'r', label='100 Days MA Closing Price')
plt.plot(df.Close, 'b', label=f'Original {User_input} Closing Price')
st.plotly_chart(fig2)

st.subheader('Closing Price vs Time chart with 100 &200MA')
fig3 = go.Figure()
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig3.add_trace(go.Scatter(x=df.reset_index()['Date'], y=df['Close'], name='stock_close'))
fig3.add_trace(go.Scatter(x=df.reset_index()['Date'], y=ma100, name='100 days MA'))
fig3.add_trace(go.Scatter(x=df.reset_index()['Date'], y=ma200, name='200 days MA'))
fig3.layout.update(title_text=f"{User_input} Close VS 100days MA", xaxis_rangeslider_visible=True, width=1000, height=500)
fig = plt.figure(figsize=(12, 6))
st.plotly_chart(fig3)

# print(df["Close"][df.reset_index()["Date"][0]])
# print(df.reset_index()['Date'])
# print(df['Close'].shape)
# splitting data into training and testing

data_traning = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])


# scale the data into a feature range for model
scaler = MinMaxScaler(feature_range=(0, 1))
data_tran_array = scaler.fit_transform(data_traning)


# Splitting the data into x_train ,y_train
time_steps = 100

x_train = []
y_train = []

for i in range(time_steps, data_tran_array.shape[0]):
    x_train.append(data_tran_array[i - time_steps:i])
    y_train.append(data_tran_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load the model
model = load_model('test_model.h5')

# testing part
past_100_days = data_traning.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(time_steps, input_data.shape[0]):
    x_test.append(input_data[i - time_steps: i])
    y_test.append(input_data[i])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test)

# index_pred = np.arange(len(y_predicted))
# index_test = np.arange(len(y_test))
# temp = y_predicted[:]
# temp1 = y_test[:]
# res_pred = {'Date':index_pred,'Close':y_predicted}
# r_p = pd.DataFrame(res_pred)

# for key in index_pred:
#     for value in temp:
#         res_pred[key] = value[0]
#         np.delete(temp, np.where(temp == value)[0][0])
#         break
#
# res_test = np.array()
# for key in index_test:
#     for value in temp1:
#         res_test[key] = value[0]
#         np.delete(temp1, np.where(temp1 == value)[0][0])
#         break
# print(res_pred)
# print(res_test)
# res_pred = dict(zip(index_pred, y_predicted))
# res_test = dict(zip(index_test, y_test))
# print(res_pred.values())

# final graph and next day prediction

submit = st.sidebar.button("Give Prediction")
if submit:
    st.subheader('Predictions vs Original')
    st.write("Model is Trained on the 70% Data set and predicting on the remaining 30% Testing data")
    # pred = go.Figure()
    pred = plt.figure(figsize=(10, 5))
    # pred.add_trace(go.Scatter(x=y_test.reshape[-1, 1], y=y_test, name='stock_close'))
    # pred.add_trace(go.Scatter(x=y_predicted.reshape[-1, 1], y=y_predicted, name='Predictions'))
    # pred.layout.update(title_text=f"{User_input} Close VS predictions", xaxis_rangeslider_visible=True)
    plt.plot(y_test, 'b', label=f'Original {User_input} Price')
    plt.plot(y_predicted, 'r', label=f'Predicted {User_input} Price')
    plt.xlabel('Time in Days', color='white')
    plt.ylabel('Price', color='white')
    plt.legend()
    st.plotly_chart(pred)

    real_data = [input_data[len(input_data) + 1 - time_steps:len(input_data)+1]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    print(real_data.shape)

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    st.subheader('Tomorrow pricing forecast')
    st.write("Tomorrow Closing Price Prediction: {0}".format(round(float(prediction), 2)))

# 30 days further prediction

further = st.sidebar.button("Give one month ahead Prediction")
if further:
    st.subheader('One month further prediction')
    st.write(f"{User_input} prediction of next month Closing price")
    future_input = input_data[len(input_data)-time_steps:]
    future_input = future_input.reshape(1, -1)
    temporary_input = list(future_input)
    temporary_input = temporary_input[0].tolist()
    # Predicting next 30 days price suing the current data
    # It will predict in sliding window manner (algorithm) with stride 1
    lst_output = []
    n_steps = 100
    i = 0
    while i < 30:
        if len(temporary_input) > 100:
            future_input = np.array(temporary_input[1:])
            future_input = future_input.reshape(1, -1)
            future_input = future_input.reshape((1, n_steps, 1))
            repeat = model.predict(future_input, verbose=0)
            temporary_input.extend(repeat[0].tolist())
            temporary_input = temporary_input[1:]
            lst_output.extend(repeat.tolist())
            i = i + 1
        else:
            future_input = future_input.reshape((1, n_steps, 1))
            repeat = model.predict(future_input, verbose=0)
            temporary_input.extend(repeat[0].tolist())
            lst_output.extend(repeat.tolist())
            i = i + 1

    # append the 30 days predicted to the graph
    # mi_new = input_data.tolist()
    # mi_new.extend(lst_output)
    # final_graph = scaler.inverse_transform(mi_new).tolist()
    lst_output = scaler.inverse_transform(lst_output)
    mi_new = y_predicted.tolist()
    mi_new.extend(lst_output)
    # Plotting final results with predicted value after 30 Days
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(y_test, 'b', label=f'Original {User_input} Price')
    plt.plot(mi_new, 'r', label=f'Predicted {User_input} Price')
    plt.xlabel('Time in Days', color='white')
    plt.ylabel('Price', color='white')
    plt.legend()
    st.plotly_chart(fig3)
    st.subheader("Closing price of the 30th day ")
    st.write('30th day Closing Price: {0}'.format(round(float(*lst_output[len(lst_output) - 1]), 2)))

# def plot_raw_data(graph):
#     figu = go.Figure()
#     figu.add_trace(go.Scatter(x=graph['Date'], y=graph['Open'], name='stock_open'))
#     figu.add_trace(go.Scatter(x=graph['Date'], y=graph['Close'], name='stock_close'))
#     figu.layout.update(title_text="Time series Data", xaxis_rangeslider_visible=True)
#     st.plotly_chart(figu)


html_string = "<a href=http://localhost:8501 >Logout</a>"
st.sidebar.markdown(html_string, unsafe_allow_html=True)
