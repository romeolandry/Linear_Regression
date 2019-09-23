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

    def loss_funtion(self, output, model):
        cost = tf.square(output - model)
        train_op = tf.train.GradientDescentOptimizer(self.__lernrate).minimize(cost)
        return train_op

    def anwendung_model (self):
        with tf.Session() as sess_pred:
           predicted_output =  sess_pred.run(self.linear_regresion_model())
           
        return predicted_output 