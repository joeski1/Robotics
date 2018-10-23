#!/usr/bin/env python
# This python file/module deals with resampling particles
# from a particle filter so we have the new particles for the next time slice
# @Author: Charlie Street

# An explanation of why I am implementing this method in terms of efficiency:
# I came to the conclusion that the theoretical best-case complexity for resampling is O(Nlog(N)).
# The reason being for this is that we have to resample particles.
# For each particle we then have to search for the correct index, i.e. particle to sample.
# This is essentially a lookup on a distribution which is best represented as a list.
# Using a slight variation on binary search gives us log(N) complexity for each particle,
# giving us O(Nlog(N)) as our complexity class.
# One could argue that using a hash table for the search would give better look up times, and it would,
# if it were not for the fact that this is
# a) not a simple lookup (we have to find the closest), and
# b) since we have to traverse the parameter (either hash table or list) earlier,
# hash table traversal is O(Nlog(N)) complexity meaning
# that would take precedence when determining our complexity class;
# it is O(log(N)) for lists and so from a design and efficiency perspective,
# the following appears to be our best option:


import random  # gonna need this for random number generation
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from noisifier import noisifier


# part_prob_list_not_norm should be in the following format:
# [((6 element particle),likelihood associated with that particle (the data likelihood))]
def resample(part_prob_list_not_norm):
    distribution = []  # list storing our cumulative distribution
    prob_sum = 0
    no_particles = len(part_prob_list_not_norm)

    part_prob_list = normalise(part_prob_list_not_norm)  # normalise the likelihoods to a probability distribution

    for i in range(len(part_prob_list)):  # loop for each probability
        prob_sum += part_prob_list[i][1]  # get the probability
        distribution.append(prob_sum)

    # due to floating point errors in normalise I am going to ensure the last element
    # of distribution is 1
    if distribution[len(distribution) - 1] != 1.0:
        distribution[len(distribution) - 1] = 1.0

    particles = []
    for _ in range(no_particles):
        new_part_rand = random.random()  # will get a number between 0 and 1
        particle_index = particle_bin_search(distribution, new_part_rand)
        noisyParticle = noisifier(part_prob_list[particle_index][0])
        particles.append(noisyParticle)

    return particles


# distribution is the cumulative distribution from the posterior probability from the particle filter
# new_part_rand is a random float between 0 and 1 which is being used to find our particle
# this is basically a slightly modified binary search
def particle_bin_search(distribution, new_part_rand):
    left = 0
    right = len(distribution) - 1
    mid = (left + right) / 2
    while left <= right:
        if new_part_rand > distribution[mid]:
            left = mid + 1
        else:
            if mid == 0:
                return 0  # due to the nature of what we are doing I can safely assume this
            elif new_part_rand > distribution[mid - 1]:  # success scenario
                return mid
            else:
                right = mid - 1
        mid = (left + right) / 2
    return -1  # error scenario, in theory should never happen

# function will normalise a set of likelihoods in form (particle, likelihood)
# into a distribution, i.e. (particle, probability)
def normalise(likelihoods):
    alpha = 0.0
    for i in range(len(likelihoods)):  # getting sum of likelihoods
        if likelihoods[i][1] > 0.0:
            alpha += likelihoods[i][1]

    normalised = []  # gonna return it rather than change it, seems neater
    counter = 0
    for i in range(len(likelihoods)):
        if likelihoods[i][1] > 0.0:
            normalised.append((likelihoods[i][0], (float(likelihoods[i][1]*100.0) / float(alpha))/100.0))
            # the *100 is there to try and help prevent floating point error
        else:
            counter += 1

    if counter > 0:
        print("Got %d negative particles" % (counter))

    return normalised
