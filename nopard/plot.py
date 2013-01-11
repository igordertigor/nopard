#!/usr/bin/env python

__doc__ = """Implements some graphical tools that are nice to have"""

import pylab as pl

def density_plot ( x, D ):
    """Plot the density D along with a confidence region"""
    # TODO: pass parameters through (e.g. color, axes, ...)
    fx = D(x)
    x_ = pl.concatenate ( (x, x[::-1]) )
    fx_ = pl.clip(pl.concatenate ( (fx+D.c,fx-D.c) ), 0, pl.inf )
    pl.fill ( x_, fx_, edgecolor=[.5]*3, facecolor=[.8]*3 )
    pl.plot ( x, fx, color=[0]*3 )

def dotplot ( x, y=0 ):
    """Create a dot plot of x. Potentially vertically shifted to y."""
    # TODO: tricky bit here is computation of the jitter...
    S = x.ptp()
    n = len(x)
    pl.plot ( x, (.2*pl.log10(n)/S * pl.randn(len(x)))+y, '.', color=[.7]*3 )

# Just copy the show function then we don't need to import pylab everytime
show = pl.show
