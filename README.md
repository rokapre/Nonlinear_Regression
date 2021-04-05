# Nonlinear Regression

This repo contains some data from my UCLA capstone project relating to a noninvasive absorbance based pH sensor. The relationship between absorbance and pH is nonlinear, and the notebooks perform the nonlinear regression via PyTorch using batch gradient descent with Adam optimizer and compare the answers to R nls() and scipy least_squares(). The Torch_NLS_pHAbs.ipynb notebook also demonstrates how a custom layer works, and how parameters of the nonlinear regression model can be regularized. New data (based on a Nonlinear Mixed Model on the original dataset) is also simulated to get train/val curves for the model built in PyTorch. 

View HTML: https://htmlpreview.github.io/?https://github.com/rokapre/Nonlinear_Regression/blob/main/Torch_NLS_pHAbs.html

