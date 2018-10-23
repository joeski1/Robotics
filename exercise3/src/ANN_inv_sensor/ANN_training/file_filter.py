#!/usr/bin/env python
# code to filter through the csv and get rid of training samples of a type we know are bad
# Author: Charlie Street

import numpy
import csv

path = 'training_data_hashes_in.csv'
write_path = 'training_data.csv'

def filter_file():
    iterations_to_check = []
    with open(path,'rb') as training_file:
        line_reader = csv.reader(training_file)
        ones = 0
        current_iteration = 0
        current_count = 0
        for row in line_reader:
            if row[0] == '#':
                if current_count <= 5:
                    iterations_to_check.append(current_iteration)
                current_iteration = row[1]
                current_count = 0
            else:
                if row[6] == '1':
                    ones = ones+1
                current_count += 1
    for i in iterations_to_check:
        print(str(i))
    print('Length: ' + str(len(iterations_to_check)) + ' ones: ' + str(ones))


def cut_out_hashes():
    '''gets rid of all the hashes in the training file'''
    with open(path,'rb') as raw_data:
        with open(write_path,'a') as to_train:
            reader = csv.reader(raw_data)
            writer = csv.writer(to_train)
            for row in reader:
                if row[0] != '#':
                    writer.writerow(row)


cut_out_hashes()

