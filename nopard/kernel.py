#!/usr/bin/env python

import numpy as np
import math
import pylab as pl

sq2pi = math.sqrt ( 2*math.pi )
sq2   = math.sqrt ( 2 )

class gaussKernel ( object ):
    def __init__ ( self ):
        pass
    def __call__ ( self, x ):
        return np.exp ( -.5*x**2 )/sq2pi
    def star ( self, x ):
        return np.exp ( -.5*(x**2/2.) )/(sq2*sq2pi) -2*self(x)

class GaussianKernelDensity1d ( object ):
    def __init__ ( self, data, alpha=0.05 ):
        """Gaussian Kernel Density estimator in one dimension

        :Parameters:
            *data*
                a 1d-array of data points
            *alpha*
                alpha level for confidence intervals (not implemented)
        """
        self.X = data
        self.n = data.shape[0]

        self.gridsize = 2**np.floor(np.log2 ( self.n )-1)

        self.alpha = alpha
        self.__fit ( data )

    def __fit ( self, data ):
        xm = data.mean()
        grid = np.mgrid[-data.ptp():data.ptp():(self.gridsize+1)*1j]
        m = grid[:-1] + .5*(grid[1]-grid[0])
        grid += xm
        n,b = np.histogram(data,grid)
        Fn = np.fft.fft(n)/self.n
        K = gaussKernel()
        Jh = 1e8
        h_ = None
        self.J_ = []

        hmin = max(data.std()/1000,3*(grid[1]-grid[0]))

        for h in np.mgrid[hmin:data.ptp():100j]:
            k = np.fft.ifftshift(K.star(m/h))
            Fk = np.fft.fft(k)
            J  = np.dot(np.fft.fft(Fk*Fn).real,n)/(self.n*h) + 2*K(0)/(self.n*h)
            if J<Jh:
                Jh = J
                h_ = h
            self.J_.append ( J )

        self.h = h_

        print h_,Jh

    def __call__ ( self, x ):
        """Evaluate the density estimator in the range of x

        TODO: This needs to be changed. We would rather like to be
        able to to either"""
        grid = np.mgrid[-x.ptp():x.ptp():(self.gridsize+1)*1j]
        m = grid[:-1]+.5*(grid[1]-grid[0])
        grid += x.mean()
        k = np.fft.ifftshift(gaussKernel()(m/self.h))
        Fk = np.fft.fft(k)
        n,b = np.histogram(self.X, grid)
        Fn = np.fft.fft(n)
        return m,np.fft.ifft(Fk*Fn).real/self.X.shape

class GaussianKernelDensity2d ( object ):
    def __init__ ( self, data, alpha=0.05 ):
        # Not working properly
        self.X = data
        self.n = data.shape[0]

        self.gridsize = 2**np.floor ( np.log2 ( self.n )-2 )

        self.alpha = alpha
        self.s = self.X.std(0)

        self.__fit ( data )

    def __fit ( self, data ):
        xm = data.mean (0)
        gridx,gridy = np.mgrid[
                -data[:,0].ptp():data[:,0].ptp():(self.gridsize+1)*1j,
                -data[:,1].ptp():data[:,1].ptp():(self.gridsize+1)*1j
                ]
        mx = gridx[:-1,:-1]
        my = gridy[:-1,:-1]
        gridx += xm[0]
        gridy += xm[1]
        n,bx,by = np.histogram2d ( data[:,0], data[:,1], bins=[gridx[:,0],gridy[0,:]] )
        Fn = np.fft.fft2 ( n )/self.n/self.n
        K = gaussKernel()
        Jh = 1e8
        h_ = None
        self.J_ = []

        hmin = .1

        for h in np.mgrid[hmin:data.ptp()/max(self.s):100j]:
            k = np.fft.ifftshift(
                    K.star(mx/(self.s[0]*h))*K.star(my/(self.s[1]*h))
                        )
            Fk = np.fft.fft2(k)
            J  = np.dot(
                np.fft.fft2(Fk*Fn).real.ravel(),
                n.ravel()
                )/(self.n*h) + 2*K(0)/(self.n*h)
            if J<Jh:
                Jh = J
                h_ = h
            self.J_.append ( J )

        self.h = h_

        print h_,Jh

    def __call__ ( self, x ):
        xm = x.mean (0)
        gridx,gridy = np.mgrid[
                -x[:,0].ptp():x[:,0].ptp():(self.gridsize+1)*1j,
                -x[:,1].ptp():x[:,1].ptp():(self.gridsize+1)*1j
                ]
        mx = gridx[:-1,:-1]
        my = gridy[:-1,:-1]
        gridx += xm[0]+gridx[1,0]-gridx[0,0]
        gridy += xm[1]+gridx[1,0]-gridx[0,0]
        n,bx,by = np.histogram2d ( self.X[:,0], self.X[:,1], bins=[gridx[:,0],gridy[0,:]] )
        Fn = np.fft.fft2 ( n )/self.n/self.n
        K = gaussKernel()
        k = np.fft.ifftshift (
                K(mx/(self.s[0]*self.h))*K(my/(self.s[1]*self.h))
                )
        Fk = np.fft.fft2(k)
        return mx,my,np.fft.ifft2 ( Fn*Fk ).real

if __name__ == "__main__":
    d1 = False
    if d1:
        X = np.random.randn ( 2000 )

        K = GaussianKernelDensity1d ( X )

        pl.subplot(211)
        pl.plot ( np.mgrid[.0001:2*X.ptp():100j],K.J_ )
        pl.subplot(212)
        pl.plot ( *(K(X)) )
        pl.show()
    else:
        X = np.random.randn(200,2)
        K = GaussianKernelDensity2d ( X )

        pl.subplot(211)
        pl.plot ( np.mgrid[.01:X.ptp()/max(K.s):100j],K.J_ )
        pl.subplot(212)
        pl.contour ( *(K(X)) )
        pl.axis('equal')
        pl.show()
