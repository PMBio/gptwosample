'''
Created on Jan 9, 2012

@author: maxz
'''

# Id for identifying the different models:
linear_covariance_model_id = "linear covariance" # Use only linear kernel for confounders
product_linear_covariance_model_id = "time x linear covariance" # Use product of time kernel and linear kernel 
reconstruct_model_id = "subtract predicted confounders" # Predict on data, from which the confounders are subtracted 
covariance_model_id = "learned confounders as covariance" # Predict on confounded data, using the predicted confounder covariance
