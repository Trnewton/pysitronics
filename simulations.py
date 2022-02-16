from multiprocessing import Pool
from abc import ABCMeta
import os, re, time
from datetime import date

import numpy as np

from pysitronics import networks as nn
from pysitronics import optimizers as op

class Job_Starter:
    def __init__(self, network_factory:nn.Network_Factory,\
                optimizer:op.Abstract_Optimizer,\
                evaluator:op.Abstract_Evaluator):
        self.network_factory = network_factory
        self.optimizer = optimizer
        self.evaluator = evaluator

    def process_func(self, params):
        g = params['g']
        q = params['q']
        sigma_seed = params['sigma_seed']
        omega_seed = params['omega_seed']

        net = self.network_factory.create_network(q=q, g=g,\
                                                sigma_seed=sigma_seed,\
                                                omega_seed=omega_seed)

        teach_steps = self.optimizer.teach_network(net)

        results = self.evaluator.test_network(net, teach_steps)
        results['g'] = g
        results['q'] = q
        results['sigma_seed'] = sigma_seed
        results['omega_seed'] = omega_seed

        return results


def parallel_QG_sweep(sweep:list, sweep_info: str,\
                        net_factory:nn.Network_Factory,\
                        optimizer:op.Abstract_Optimizer,\
                        evaluator:op.Abstract_Evaluator,\
                        save_dir:str, processes:int=12):
    ''''''

    # Start timer
    tic = time.perf_counter()

    #################### Run grid sweep ####################
    # Create pickleable job starter
    job_starter = Job_Starter(network_factory=net_factory, optimizer=optimizer,\
                            evaluator=evaluator)

    # Create process pool and map jobs to processes
    pool = Pool(processes=processes)
    results = pool.map(job_starter.process_func, sweep)


    #################### Save Data ####################
    dir_path = save_dir
    if not os.path.exists(dir_path):
            print('Directory:', dir_path, '\ndoes not exist, creating new one.')
            os.makedirs(dir_path)

    # Check if file exists and add copy number so as not to overwrite,
    run_name = 'Run_1/'
    copy = 2
    while os.path.exists(dir_path + run_name):
            run_name = '_'.join(re.split('_', run_name)[:-1]) + f'_{copy}/'
            copy += 1

    dir_path += run_name

    print('Saving to:', dir_path)
    os.makedirs(dir_path)

    # Create description file
    with open(dir_path + 'Description.txt', mode='w') as desc_file:
            desc_file.write(f'Date:{date.today()}\n')
            # Sweep
            desc_file.write(sweep_info +'\n')
            # Network
            desc_file.write(net_factory.to_string())
            # Optimizer
            desc_file.write(f'Optimizer:{optimizer.to_string()}\n')
            # Evaluator
            desc_file.write(f'Evaluator:{evaluator.to_string()}')

    evaluator.save_results(results, dir_path)

    ######## End timer ########
    toc = time.perf_counter()
    time_dif = toc - tic
    s = time_dif % 60
    m = (time_dif % (60*60)) // 60
    h = (time_dif % (60*60*24)) // (60*60)
    d = time_dif // (60*60*24)

    print(f'Done. Total time: {d}:{h:2.0f}:{m:2.0f}:{s:2.0f}')

def series_QG_sweep(sweep:list, sweep_info: str, net_factory:nn.Network_Factory,\
        optimizer:op.Abstract_Optimizer, evaluator:op.Abstract_Evaluator, save_dir:str):
	# Start timer`
	tic = time.perf_counter()

	#################### Run grid sweep ####################
	# Create pickleable job starter
	job_starter = Job_Starter(network_factory=net_factory, optimizer=optimizer, evaluator=evaluator)

	# Create process pool and map jobs to processes
	results = []
	for params in sweep:
		results.append(job_starter.process_func(params))


	#################### Save Data ####################
	dir_path = save_dir
	if not os.path.exists(dir_path):
			print('Directory:', dir_path, '\ndoes not exist, creating new one.')
			os.makedirs(dir_path)

	# Check if file exists and add copy number so as not to overwrite,
	run_name = 'Run_1/'
	copy = 2
	while os.path.exists(dir_path + run_name):
			run_name = '_'.join(re.split('_', run_name)[:-1]) + f'_{copy}/'
			copy += 1

	dir_path += run_name

	print('Saving to:', dir_path)
	os.makedirs(dir_path)

	# Create description file
	with open(dir_path + 'Description.txt', mode='w') as desc_file:
			desc_file.write(f'Date:{date.today()}\n')
			# Sweep
			desc_file.write(sweep_info +'\n')
			# Network
			desc_file.write(net_factory.to_string())
			# Optimizer
			desc_file.write(f'Optimizer:{optimizer.to_string()}\n')
			# Evaluator
			desc_file.write(f'Evaluator:{evaluator.to_string()}')

	evaluator.save_results(results, dir_path)

	######## End timer ########
	toc = time.perf_counter()
	time_dif = toc - tic
	s = time_dif % 60
	m = (time_dif % (60*60)) // 60
	h = (time_dif % (60*60*24)) // (60*60)
	d = time_dif // (60*60*24)

	print(f'Done`. Total time: {d}:{h:2.0f}:{m:2.0f}:{s:2.0f}')