#!/usr/bin/env python

import numpy as np
import math

sq2pi = math.sqrt ( 2*math.pi )
sq5   = math.sqrt ( 5. )

def gaussKernel ( x ):
    return np.exp ( -0.5*x**2 )/sq2pi

def epanechikovKernel ( x ):
    return np.clip(.75*(1-.2*x**2)/sq5,0,1)

class kernelDensity ( object ):
    def __init__ ( self, data, kernel, alpha=0.05 ):
        self.X = data
        self.kernel = kernel
        self.alpha = alpha
        self.__fit ( data )
    def __fit ( self, data ):
        pass
