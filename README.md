# Homework

Refer to some online courses, paper


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

## GAN
Download datasets with
`tf.keras.datasets.mnist.load_data()`

generator


    model = tf.keras.Sequential()  
    model.add(layers.Dense(256,input_shape=(100,),use_bias=False))  
    model.add(layers.BatchNormalization())  
    model.add(layers.LeakyReLU())  
    
    model.add(layers.Dense(512,use_bias=False))  
    model.add(layers.BatchNormalization())   
    model.add(layers.LeakyReLU())  
    
    model.add(layers.Dense(28*28*1,use_bias=False,activation='tanh'))  
    model.add(layers.BatchNormalization())  
    
    model.add(layers.Reshape((28,28,1)))  
    
discriminator

    model = keras.Sequential()
    
    model.add(layers.Flatten())              
    
    model.add(layers.Dense(512,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(256,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(1))
    
discriminator_loss

    read_loss =cross_entropy(tf.ones_like(real_image),real_image)    
    fake_loss =cross_entropy(tf.zeros_like(fake_out),fake_out)       
    
generator_loss

    cross_entropy(tf.ones_like(fake_out),fake_out) 

training

![Webp net-gifmaker](https://user-images.githubusercontent.com/96108439/154017695-f927b4f6-93a8-464d-9a41-9f9f16cd34c3.gif)


## VAE

Download datasets with `tf.keras.datasets.mnist.load_data()`

MODEL


     def __init__(self):
        super(VAE_model, self).__init__()        
        self.lin_1 = tf.keras.layers.Dense(400)
        self.lin_2 = tf.keras.layers.Dense(20)
        self.lin_3 = tf.keras.layers.Dense(20)
        self.lin_4 = tf.keras.layers.Dense(400)
        self.lin_5 = tf.keras.layers.Dense(784)
        
    def encode(self, x):                                       
        h1 = tf.nn.relu(self.lin_1(x))                
        return self.lin_2(h1), self.lin_3(h1)
    
    def reparameters(self, mu, logvar):
        std = tf.math.exp(0.5*logvar)
        eps = tf.random.normal(std.shape, mean=0.0, stddev=1.0)   
        return mu + eps*std
        
    def decode(self, z):
        h4 = tf.nn.relu(self.lin_4(z))                   
        return tf.sigmoid(self.lin_5(h4))
    
    def call(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameters(mu, logvar)
        return self.decode(z), mu, logvar

loss_function

    BCE_loss =tf.keras.losses.binary_crossentropy(x,recon_x)
    KLD_loss=-0.5*tf.reduce_sum(1+logvar-tf.pow(mu,2)-tf.exp(logvar))
    return BCE_loss+0.0001*KLD_loss  
    
training

![Webp net-gifmaker (2)](https://user-images.githubusercontent.com/96108439/154030194-cf5718dd-7ca5-4d93-89fd-4636c65f2c72.gif)


    
