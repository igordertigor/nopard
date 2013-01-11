#!/usr/bin/env python

__doc__ = """Implements histograms with optimal bin width as discussed
in the book by Wasserman: All of statistics. Note that the Formula for the
crossvalidated risk is wrong in that book and has been taken from
Wasserman: All of nonparametric statistics."""

import numpy as np
import math
import scipy.optimize as opt
import scipy.stats as stats

class histogramDensity ( object ):
    def __init__ ( self, data, interval=(0,1), alpha=0.05 ):
        """Histogram density estimation with optimal bin width

        :Parameters:
            *data*
                1d array of observations
            *interval*
                start and endpoint of the interval on which the data live
            *alpha*
                create 1-alpha confidence regions
        """
        self.x0 = float(interval[0])
        self.interval_length = float(interval[1] - interval[0])
        self.alpha = alpha
        self.__fit ( data )

    def __fit ( self, data ):
        n = len(data)
        def J(h):
            h = h**2
            nbins = math.ceil ( self.interval_length/h )
            nj,bins = np.histogram ( data, self.x0 + h*np.arange ( nbins+1 ) )
            pj = nj.astype('d')/np.sum(nj)
            risk = (2. - float(n+1) * np.sum ( pj**2 ))/(h*(n-1))
            return risk
        self.h = opt.brent ( J, brack=(
            math.sqrt(self.interval_length/math.sqrt(n)),
            math.sqrt(self.interval_length)) )
        self.h = self.h**2
        self.nbins = math.ceil ( self.interval_length/self.h )
        nj,self.bins = np.histogram ( data, self.x0+self.h*np.arange ( self.nbins+1 ) )
        self.pj = nj.astype('d')/np.sum(nj)
        self.bins.shape = (-1,1)
        self.c = 0.5*stats.norm.ppf(self.alpha/(2*self.nbins)) * math.sqrt ( self.nbins/n )

    def __call__ ( self, x ):
        x = x.reshape ( (1,-1) )
        i = np.where ( np.logical_and ( x < self.bins[1:], self.bins[:-1] <= x) ) [0]
        return self.pj[i]/self.h

if __name__ == "__main__":
    import plot
    data = np.random.randn ( 4000 )
    H = histogramDensity ( data, (-4,4) )
    x = np.mgrid[-4:4:100j]
    plot.density_plot ( x, H )
    plot.show()
