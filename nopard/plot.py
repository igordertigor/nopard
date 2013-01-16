#!/usr/bin/env python

__doc__ = """Implements some graphical tools that are nice to have"""

import pylab as pl

def density_plot ( x, D ):
    """Plot the density D along with a confidence region"""
    # TODO: pass parameters through (e.g. color, axes, ...)
    fx = D(x)
    x_ = pl.concatenate ( (x, x[::-1]) )
    fx_ = pl.clip(pl.concatenate ( (fx+D.c,fx[::-1]-D.c) ), 0, pl.inf )
    pl.fill ( x_, fx_, edgecolor=[.5]*3, facecolor=[.8]*3 )
    pl.plot ( x, fx, color=[0]*3 )

def dotplot ( x, y=0 ):
    """Create a dot plot of x. Potentially vertically shifted to y."""
    # TODO: tricky bit here is computation of the jitter...
    S = x.ptp()
    n = len(x)
    pl.plot ( x, (.2*pl.log10(n)/S * pl.randn(len(x)))+y, '.', color=[.7]*3 )

def qqplot ( th, D, interval=(0,1) ):
    """Create a q-q plot of theta with respect to density D

    :Parameters:
        *th*
            a sample that might be from the density D
        *D*
            a density object
    """
    th = th.copy()
    th.sort()
    nq = len(th)
    q = (pl.arange ( 0, nq, dtype='d' )+1)/nq
    x = pl.mgrid[interval[0]:interval[1]:1j*4*nq]
    f = D(x)
    qD = pl.cumsum(f)*pl.diff(x)[0]/pl.trapz(f,x)
    th_ = []
    for q_ in q:
        i = pl.where ( qD<=q_ )[0]
        th_.append ( x[i[-1]] )

    pl.plot ( th, th_, '.' )
    pl.plot ( [th[0],th[-1]],[th[0],th[-1]], 'k:' )
    # pl.plot ( th, q )
    # pl.plot ( x, qD )

# Just copy the show function so that we don't need to import pylab everytime
show = pl.show
