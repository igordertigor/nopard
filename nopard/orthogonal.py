#!/usr/bin/env python

__doc__ = """Density estimation using orthogonal basis functions as described
in Wasserman: All of statistics.
"""

import numpy as np
from scipy import stats
from scipy import special

twopi = (2*np.pi)

class orthogonalBasis ( object ):
    """Base class for orthogonal bases"""
    def __call__ ( self, j, x ):
        pass

class LegendreBasis ( orthogonalBasis ):
    """basis of legendre polynomials"""
    def __init__ ( self ):
        self.K = 2
    def __call__ ( self, j, x ):
        if j>40:
            print "WARNING: legendre polynomials for j>40 are numerically unstable"
        P = np.poly1d ( special.legendre ( j ) )
        return P(2*x-1)

class CosineBasis ( orthogonalBasis ):
    """Cosine basis Phi_j = sqrt(2)*cos(pi*j*x)"""
    def __init__ ( self ):
        self.sq2 = np.sqrt(2.)
        self.K = self.sq2
    def __call__ ( self, j, x ):
        return self.sq2 * np.cos ( j*np.pi*x )
    def __str__ ( self ):
        return "<CosineBasis>"

class FourierBasis ( orthogonalBasis ):
    """Fourier basis

    Phi_j = 2cos(2pi*j*x), j=0,1,3,5,...
    Phi_j = 2sin(2pi*j*x), j=2,4,6,8,...
    """
    def __init__ ( self ):
        self.sq2 = np.sqrt(2.)
        self.K = 2.
    def __call__ ( self, j, x ):
        if j == 0:
            return 0.*x+1
        elif j%2 == 0:
            return 2*np.sin ( twopi*j*x )
        else:
            return 2*np.cos ( twopi*j*x )
    def __str__ ( self ):
        return "<FourierBasis>"

class orthogonalDensity ( object ):
    def __init__ ( self, data, basis, interval=(0,1), alpha=0.05, data_are_weights=False ):
        """Perform orthogonal basis function density estimation

        :Parameters:
            *data*
                1d array of observations
            *basis*
                an orthogonalBasis object
            *interval*
                interval on which the density is assumed to live
            *alpha*
                determine 1-alpha confidence regions
            *data_are_weights*
                interprete the data array as the weights

        Calling the resulting density object evaluates the estimated density.
        """
        self.x0 = interval[0]
        self.interval_length = float(interval[1]-interval[0])
        self.scale = 1./self.interval_length
        self.alpha = alpha
        self.Phi = basis
        self.Z = 1.

        if data_are_weights:
            self.b = data
            self.jmax = len(data)
        else:
            nmax = int(np.ceil(np.sqrt(len(data))))
            self.b = np.zeros ( nmax, 'd' )
            self.sg = np.zeros ( nmax, 'd' )
            self.nmax = nmax
            self.R = np.zeros ( nmax, 'd' )
            self.__fit( self.scale*(data-self.x0) )

        if not isinstance ( basis, orthogonal2d ):
            self.__normalize1d ()
        else:
            self.__normalize2d ()

    def __fit ( self, data ):
        n = len(data)
        for j in xrange ( self.nmax ):
            Phi_j = self.Phi( j, data )
            # this estimates the parameter
            self.b[j] = Phi_j.mean()
            # this is part of a shortcut formula for the crossvalidated risk
            self.sg[j] = np.sum((Phi_j - self.b[j])**2)/(n-1)

        for j in xrange ( self.nmax ):
            # use shortcut formula for crossvalidated risk
            self.R[j] = self.sg[:j].sum()/n + np.sum( np.clip(self.b[j:]**2-self.sg[j:]/n,0,1e10))
        self.jmax = np.argmin(self.R)

        # size of confidence region
        self.c = self.Phi.K**2 * \
                np.sqrt ( self.jmax*stats.chisqprob ( self.alpha, self.jmax )/n )

    def __normalize1d ( self ):
        # Perform numerical integration to normalize the nonnegative density
        x = np.mgrid[self.x0:self.interval_length:100j]
        f = self(x)
        self.Z = np.trapz ( f, x )
    def __normalize2d ( self ):
        x,y = np.mgrid[self.x0:self.interval_length:100j,self.x0:self.interval_length:100j]
        f = self(np.c_[x.ravel(),y.ravel()]).reshape(x.shape)
        self.Z = np.trapz ( np.trapz (f,x[:,0],axis=0),y[0,:])

    def __call__ ( self, x ):
        """Evaluates the density at position x"""
        out = np.zeros ( x.shape[0], 'd' )
        for j in xrange ( self.jmax ):
            out += self.b[j]*self.Phi(j, self.scale*(x-self.x0) )
        return np.clip ( out/self.scale/self.Z, 0, 1e10 )

    def __str__ ( self ):
        return "<orthogonalDensity on (%g,%g), basis: %s, J^=%g>" % \
                (self.x0,self.x0+self.interval_length,str(self.Phi),self.jmax)

# 2d

class orthogonal2d ( orthogonalBasis ):
    def __init__ ( self, Basis1d ):
        self.basis1d = Basis1d
        self.K = self.basis1d.K
    def __call__ ( self, j, x ):
        k,l = self.__map_j_to_kl ( j )
        return self.basis1d ( k,x[:,0] )*self.basis1d ( l,x[:,1] )
    def __map_j_to_kl ( self, j ):
        k = l = 0
        d = 1
        count = 0
        while count < j:
            for k in xrange ( d+1 ):
                l = d-k
                count += 1
                if count == j:
                    break
            d += 1
        return k,l
    def __str__ ( self ):
        return "ortogonal2d (%s)" % (str(self.basis1d))

if __name__ == "__main__":
    import plot
    data = np.concatenate ( [np.random.vonmises ( 1, 5, size=(2000,)),
        np.random.vonmises(0, 10, size=(4000,) ),
        np.random.vonmises(0, .01, size=(2000,) )
        ] )
    D = orthogonalDensity ( data, FourierBasis(), (-np.pi,np.pi) )
    print D

    x = np.mgrid[-np.pi:np.pi:100j]
    plot.density_plot ( x, D )
    plot.show()
