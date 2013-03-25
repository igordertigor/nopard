nopard
======

NOn PARametric Density modeling

Occasionally, it turns out to be useful to get an idea of the density of some
experimentally observed data. Sure, there is pylab.hist, or scipy.stats.kde. Yet, the "smoothing" parameter that plays such a central role in nonparametric statistics is completely ignored by these tools. For many standard tools in nonparametric density estimation, there are also standard formulas for leave-one-out cross-validated risk. These are implemented here. This is not meant to be the ultimate density modeling tool -- in fact standard R is much better at this -- but it owes to the fact, that I observed myself either reimplementing these procedures over and over in python, or that I did *not* use cross-validated risk to set my smoothing parameters. So I decided to write **nopard** like **No** n **par** ametric **d** ensity modeling.
