import numpy as np

def calculate_dW_hat(L,theta,Js_hat,kesi_x,kesi_rall,J_pinv,kesi_q,kesi_x_prime):
    dW_hat = - L @ theta @ (Js_hat.T @ kesi_x + kesi_rall + J_pinv.T @ kesi_q).T
    dW_hat = dW_hat @ kesi_x_prime
    return dW_hat