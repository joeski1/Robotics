#!/usr/bin/env python
# this python file/module is used to implement kld sampling
# kld sampling is used in order to perform an adaptive form
# of Monte Carlo Localisation
# By adaptive, we mean that the number of particles decreases as we get more sure
# of our position, since the 'bins' associated with certain positions will increase drastically
# as there is a higher posterior belief associated with them
# all of the logic for the kld sampling algorithm was gained from the following paper
# and I give credit to the author here also:
# Title: 'KLD-Sampling: Adaptive Particle Filters and Mobile Robot Localisation'
# Author: Dieter Fox of the Department of Computer Science & Engineering at
# the University of Washington' Published: September 19, 2001
# @author: Charlie Street

# need mathematical  and random stuff
import math
import random
from scipy.special import erfinv

# allows me to import from my files in the same directory
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# importing reusable functions from resampler
from resampler import particle_bin_search
from resampler import normalise
from noisifier import noisifier

# method will find the normal distribution quantile function value for p
# found out equation from https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function
def probit_function(p):
    return math.sqrt(2) * erfinv((2*p)-1)


ERROR = 0.285 # the threshold for the difference between the maximum likelihood threshold and the posterior
SIGMA = 0.5 # 1 - p(K(maxP,P) < ERROR) must be 0<=SIGMA<=1 (but either of the bounds would be silly)
z_one_minus_sigma = probit_function(1-SIGMA)  # i actually only need to calculate this once!!!
nMin = 100  # a user defined value which sets the minimum number of particles to generate
nMax = 500  # a user defined value which sets the minimum number of particles to generate


# this function approximates the value of the
# chi squared distribution for values n, k, ERROR, and SIGMA
# to find out more information about the derivation of this
# please reference the paper mentioned at the top of this file
# in particular page 7
def chi_square_approx(k):

    k_over_error = float(k-1)/float(2*ERROR)
    over_k = float(2)/float(9*(k-1))
    root_bit = math.sqrt(over_k)
    inside_brackets = 1 - over_k + (root_bit * z_one_minus_sigma)
    cubed = math.pow(inside_brackets, 3)
    final_approx = k_over_error * cubed
    return final_approx


# this function will have a reasonably similar layout to that of resample
# in resampler.py The difference here is that rather than generating the same
# number of samples as there were previously, we now change this depending on
# the number of bins and their weights etc.
def kld_sample(likelihoods):
    distribution = []  # going to make our distribution first
    dist_sum = 0

    prob_from_data = normalise(likelihoods)  # normalise our data

    print("len(likelihoods) = %d, len(prob_from_data) = %d" % (len(likelihoods), len(prob_from_data)))

    #if len(prob_from_data) == 0:
    #    return []

    for i in range(len(prob_from_data)):
        dist_sum += prob_from_data[i][1]
        distribution.append(dist_sum)

    # due to floating point errors in normalise I am going to ensure the last element
    # of distribution is 1
    if distribution[len(distribution)-1] != 1:
        distribution[len(distribution)-1] = 1

    # these values are important because they allow us to determine the new sample set size
    n = 0  # number of particles in new sample set
    k = 0  # number of bins used

    empty_bins = []  # need to make something to specify whether a bin is empty or not
    # a bin in this case is an index of distribution
    for i in range(len(distribution)):
        empty_bins.append(True)

    particles = []  # the array to store our particles in

    finished = False  # loop condition to emulate do while loop
    while not finished:
        new_particle = random.random()  # a number between 0 and 1 to select a bin
        part_index = particle_bin_search(distribution, new_particle)  # get the bin index
        noisy_particle = noisifier(prob_from_data[part_index][0])
        particles.append(noisy_particle)  # add to new sample set
        if empty_bins[part_index]:  # if this is the first particle in the bin, incrememnt k
            k += 1
            empty_bins[part_index] = False  # this bin is not empty anymore

        n += 1

        if k > 1:
            finished = (not (n < chi_square_approx(k))) and (not (n < nMin))  # the condition in our pseudo do-while loop
        else:
            finished = not (n < nMin)

        if len(particles) >= nMax:
            finished = True
    return particles
