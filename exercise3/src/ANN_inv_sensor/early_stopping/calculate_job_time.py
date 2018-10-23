#!/usr/bin/env python3
import sys

def time_for_job(epochs):
    # running 10 folds with full cpu load
    #return 15.941939639329911*epochs + 72.70859847068641
    # running 8-folds with low cpu load
    #return 6.476327231952123*epochs + 149.2679733662378
    # running 8 folds with full cpu load
    return 11.298862204063507*epochs + 81.93789477007522

def low_load_mult(num_jobs, job_num, num_cores):
    jobs_left = num_jobs - job_num
    if jobs_left >= num_cores:
        return 1
    else:
        return 0.75 # roughly

RAW_TIME=False
def fmt_time(secs):
    if RAW_TIME:
        return str(secs)
    else:
        hours, remainder = divmod(secs, 60*60)
        minutes, seconds = divmod(remainder, 60)
        return '{}h{:02}'.format(int(hours), int(minutes))


def print_estimate(min_e, max_e, step_e, num_cores, last_multiplier):
    epochs = list(range(min_e, max_e+1, step_e))
    num_jobs = len(epochs)
    job_times = [time_for_job(epochs) for epochs in epochs]

    if last_multiplier:
        # get multiplier for the last few jobs
        m = lambda i: low_load_mult(num_jobs, i, num_cores)
        job_times = [t*m(i) for i,t in enumerate(job_times)]
        print('with last jobs multiplier')

    total_time = sum(job_times)
    average_time = total_time / num_jobs

    estimated_time = total_time/min(num_cores, num_jobs)

    print('{} jobs with times:\n{}'.format(
        num_jobs, ', '.join([fmt_time(t) for t in job_times])))
    print('total serial time: {}, average serial time: {}'.format(
        fmt_time(total_time), fmt_time(average_time)))
    print('estimated time: {} (total of {} seconds)'.format(
        fmt_time(estimated_time), estimated_time))

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('usage: ./calculate_job_time.py min max step')
        sys.exit(1)

    min_e, max_e, step_e = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    num_cores = 8

    print('min {} max {} step {} epochs on {} cores'.format(
        min_e, max_e, step_e, num_cores))

    print_estimate(min_e, max_e, step_e, num_cores, False)
    #print()
    #print_estimate(min_e, max_e, step_e, num_cores, True)

