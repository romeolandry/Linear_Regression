import tensorflow as tf 


class LinearModel:
    def __init__ (self,eingabe, gewicht, bias, lernrate):
        self.__lernrate = lernrate
        self.__eingabe = eingabe
        self.__gewicht = gewicht
        self.__bias = bias

    def update_model (self,gewicht,bias):
        self.__gewicht = gewicht
        self.__bias = bias
            
    def linear_regresion_model(self):
        return (tf.add(tf.multiply(self.__eingabe,self.__gewicht,),self.__bias)) ## model = x*w + b

    def loss_funtion(self, Y):
        cost = tf.square(Y - self.linear_regresion_model)
        train_op = tf.train.GradientDescentOptimizer(self.__lernrate).minimize(cost)
        return train_op

    def train_model(self, epoch,input_daten,output_daten, out_funktion,input_model,output_model):
        with tf.Session() as sess_train:
            for i in range(epoch):
                print("----------Epoch: {} ---".format(i))
                for (x,y) in zip(input_daten,output_daten):
                    sess_train.run(self.loss_funtion(out_funktion),feed_dict={input_model:x,output_model:y})

                # Berechnung von Gewichtungen und Bias
                weight_value = sess_train.run(self.__gewicht)
                bias_value = sess_train.run(self.__bias)

                print("bias_value :{}".format(bias_value))
                print("weight_value:{}".format(weight_value))

            return (weight_value,bias_value)

    def anwendung_model (self,test_input,weight_value,bias_value):
        predict_model = tf.add(tf.multiply(test_input,weight_value),bias_value)
        with tf.Session() as sess_pred:            
            predicted_output = sess_pred.run(predict_model)
        return predicted_output 