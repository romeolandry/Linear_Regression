import tensorflow as tf 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg') # ?
import matplotlib.pyplot as plt

np.random.seed(42)

class DatenGenerierung:
    def __init__(self,my_weight,range_input):
        self.__weight = my_weight
        self.__input = np.arange(0,range_input,0.1)
        self.__noise = np.random.randint(low=-5, high=5, size=self.__input.shape)

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