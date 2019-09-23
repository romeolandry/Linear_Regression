import DatenGenerierung
import LinearModel
import tensorflow as tf
import numpy as np 


## Variable definition
lernrate = 0.001

X = tf.placeholder(tf.float32, name="X")# input des Models
Y = tf.placeholder(tf.float32, name="Y") # output des Models

w = tf.Variable(0.0,name="weigthts")
b = tf.Variable(0.0,name="bias")

# Daten Generierung

input_array = (0,15,0.1)
daten = DatenGenerierung.DatenGenerierung(4,input_array)
input_val,output_val = daten.gen_daten()

#daten.data_visualisation(input_val,output_val)

# Training des Models
klasse_Model = LinearModel.LinearModel(X,w,b,lernrate)
train_model = klasse_Model.linear_regresion_model()
#------------------- set cost 
train_op = klasse_Model.loss_funtion(Y)
# training
weight_value, bias_value = klasse_Model.train_model(100,input_val,output_val,X,Y,train_op)

## test des Models
test_input = np.arange(*input_array)
prediction = klasse_Model.anwendung_model(test_input,weight_value,bias_value)

# show result
daten.show_result("Funktion",input_val,output_val,test_input,prediction,weight_value,bias_value)





