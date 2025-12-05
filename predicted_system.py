# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 08:39:35 2025

@author: Anshika Agarwal
"""

import numpy as np
import pickle
loaded_model=pickle.load(open('C:/Users/Anshika Agarwal/Documents/streamlit/heart disease prediction/trained_model.sav','rb'))
input_data=[58,1,0,150,270,0,0,111,1,0.8,2,0,3]
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped) 
print(prediction)
if(prediction[0]==1):
  print("heart disease")
else:
  print("healty heart")
  