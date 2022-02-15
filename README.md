# Homework


## LSTM
Predict the PM2.5

[Data](123.csv)

data preprocessing

Before

<img src="https://user-images.githubusercontent.com/96108439/154005280-a5b77fbd-eea4-4e1b-aeee-17a9d372f467.PNG" alt="drawing" width="500px"/>

after

<img src="https://user-images.githubusercontent.com/96108439/154005366-a0122585-7f38-47c4-ad40-878a0d0ffcc0.PNG" alt="drawing" width="500px"/>

Model

`model=keras.Sequential()`  
 `model.add(tf.keras.layers.LSTM(32,input_shape=(X_train.shape[1:]),return_sequences=True))`  
`model.add(tf.keras.layers.LSTM(32,return_sequences=True))`  
`model.add(tf.keras.layers.LSTM(32))`  
`model.add(tf.keras.layers.Dense(1))`  

Result
150 Epochs  
<img src="https://user-images.githubusercontent.com/96108439/154009691-f443c2fe-b458-4095-879d-27148ed37ec2.PNG" width="500px"/>

