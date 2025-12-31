# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 06:23:51 2025

@author: uig67136
"""

import numpy as np

class KalmanFilterCV:
    def __init__(self, dt=1.0):
        self.dt = dt

        # State transition matrix
        # self.F = np.array([
        #     [1, dt],
        #     [0, 1 ]
        # ])
        # State transition matrix with acceleration
        self.F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        # Measurement matrix (we directly observe R and v)
        #self.H = np.eye(2)
        #measrement matrix with 0 for accel since we measure only R & v
        self.H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])

        # Process noise (model uncertainty)
        # self.Q = np.array([
        #     [0.1, 0.0],
        #     [0.0, 0.1]
        # ])
        #q = 0.1
        #self.Q = q * np.eye(3)
        sigma_a = 5.0  # acceleration noise (tune this)
        dt = self.dt
        
        self.Q = sigma_a**2 * np.array([
            [dt**4/4, dt**3/2, dt**2/2],
            [dt**3/2, dt**2,   dt],
            [dt**2/2, dt,      1]
        ])


        # Measurement noise (sensor uncertainty)
        self.R = np.array([
            [0.5, 0.0],
            [0.0, 0.5]
        ])

        #self.x = None  # state [R, v]
        #self.P = None  # state covariance
        
        # State covariance
        self.P = np.eye(3) * 10

        # State vector
        self.x = np.zeros((3, 1))
    # def init_state(self, R0, v0):
    #     self.x = np.array([[R0],
    #                        [v0]])
    #     self.P = np.eye(2) * 1.0
        
    def init_state(self, R0, v0, a0=0.0):
       self.x = np.array([[R0], [v0], [a0]])
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    # def update(self, z):
    #     z = z.reshape(2,1)
    #     y = z - self.H @ self.x               # innovation
    #     S = self.H @ self.P @ self.H.T + self.R
    #     K = self.P @ self.H.T @ np.linalg.inv(S)

    #     self.x = self.x + K @ y
    #     self.P = (np.eye(2) - K @ self.H) @ self.P
        
    def update(self, z):
        z = z.reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P
