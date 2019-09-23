import tensorflow as tf 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg') # ?
import matplotlib.pyplot as plt

np.random.seed(42)

class DatenGenerierung:
    def __init__(self,my_weight,range_input):
        self.__weight = my_weight
        self.__input = np.arange(*range_input)
        self.__noise = np.random.randint(low = -2,high=2 ,size=self.__input.shape)

    #getter und setter
    def get_weight(self):
        return self.__weight
    def set_weight(self,my_weight):
        self.__weight = my_weight

    def get_input(self):
        return self.__input
    def set_input(self,input_value):
        self.__input = np.arange(input_value)
    
    def get_noise(self):
        return self.__noise
    def set_noise(self,noise):
        self.__noise = noise
    
    #Daten Generieren

    def gen_daten(self):
        output = self.__weight * self.__input + self.__noise
        return (self.__input,output)

    def data_visualisation(self, input_val, output):
        plt.scatter(input_val, output, c= "red")
        plt.show()

    def show_result (self,title,input_val,outpu_val,input_test,prediction,weight_value,bias_value):
        
        plt.title(title +' y= X*w + b  mit w= '+ str(weight_value) + 'und b = '+ str(bias_value))
        plt.scatter(input_val, outpu_val, c= "red", s=4, label="Original Werte")
        plt.scatter(input_test, prediction, c= "blue", s=5, label="Vorhergesagte Werte")
        plt.legend(loc='upper left')
        plt.savefig('linearregression.png')

        print("Vorhergesagewert für w: " + str(weight_value))

        plt.show(block=True)
