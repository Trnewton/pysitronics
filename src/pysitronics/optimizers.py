from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from numpy.linalg import inv

from pysitronics import networks as nn


class Abstract_Optimizer(ABC):
    ''''''

    @abstractmethod
    def teach_network(self, network:nn.Abstract_Network) -> None:
        ''''''

    @abstractmethod
    def to_string(self) -> str:
        '''Expresses type and params of optimizer as a string for recording.'''

class FORCE(Abstract_Optimizer):
    def __init__(self, alpha, supervisor, dt, warmup_steps, training_steps, learn_step) -> None:
        super().__init__()

        self.alpha = alpha
        self.sup = supervisor
        self.dt = dt
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.learn_step = learn_step

    def teach_network(self, network: nn.Abstract_Network) -> int:

        # PreTraining network run
        _ = network.simulate(self.dt, self.warmup_steps)

        P = self.alpha * np.eye(network.N)
        c = 0
        k = 0

        for n in range(self.training_steps):
            z, R = network.step(self.dt)
            if (n % self.learn_step) == 0:
                signal = self.sup[n%self.sup.shape[0]]
                error = z - signal

                k = np.dot(R, P)
                rPr = np.dot(R, k)
                c = 1.0/(1.0 + rPr)

                # Update P
                P = P - c * np.outer(k, k)

                # Compute change in readout weight vector
                if network.dim == 1:
                    dphi = - c * error * k
                else:
                    dphi = - c * np.outer(error, k)

                network.set_decoder(network.get_decoder() + dphi)

        return self.training_steps

    def to_string(self) -> str:
        return f'\n\tdt:{self.dt}'\
            + f'\n\talpha:{self.alpha}'\
            + f'\n\twarmup:{self.warmup_steps}'\
            + f'\n\ttrain_steps:{self.training_steps}'\
            + f'\n\tlearn_step:{self.learn_step}'

class FORCE_Verbose(Abstract_Optimizer):
    def __init__(self, alpha, supervisor, dt, warmup_steps, training_steps, learn_step) -> None:

        self.alpha = alpha
        self.sup = supervisor
        self.dt = dt
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.learn_step = learn_step

    def teach_network(self, network: nn.Abstract_Network) -> int:

        # PreTraining network run
        _ = network.simulate(self.dt, self.warmup_steps)

        self.dim = self.sup.shape[1] if len(self.sup.shape) == 2 else 1
        z_series = np.zeros((self.training_steps, self.dim)).squeeze()

        P = self.alpha * np.eye(network.N)
        c = 0
        k = 0

        for n in range(self.training_steps):
            z, R = network.step(self.dt)
            z_series[n] = z

            if (n % self.learn_step) == 0:
                signal = self.sup[n%self.sup.shape[0]]
                error = z - signal

                k = np.dot(R, P)
                rPr = np.dot(R, k)
                c = 1.0/(1.0 + rPr)

                # Update P
                P = P - c * np.outer(k, k)

                # Compute change in readout weight vector
                if network.dim == 1:
                    dphi = - c * error * k
                else:
                    dphi = - c * np.outer(error, k)

                network.set_decoder(network.get_decoder() + dphi)

        return self.training_steps, z_series

    def to_string(self) -> str:
        return f'\n\tdt:{self.dt}'\
            + f'\n\talpha:{self.alpha}'\
            + f'\n\twarmup:{self.warmup_steps}'\
            + f'\n\ttrain_steps:{self.training_steps}'\
            + f'\n\tlearn_step:{self.learn_step}'

def echo_state(dt, alpha, sup, net:nn.Abstract_Network, i_in=None):

    dim = sup.shape[1] if len(sup.shape) > 1 else 1
    RR = np.zeros((net.N, net.N))
    Rx = np.zeros(dim)

    for i, sup_i in enumerate(sup):
        z, R_i = net.step(dt)
        RR = np.outter(R_i, R_i)
        net.z = sup_i

    phi = np.matmul(inv(RR + (1/alpha) * np.eye(net.N)), sup)

    return phi

class Abstract_Evaluator(ABC):
    ''''''

    @abstractmethod
    def test_network(self, network:nn.Abstract_Network, current_step:int) -> Dict:
        '''Tests the training of the given network and returns the results as a
            dictionary to be collected and made into a pandas dataframe.'''

    @abstractmethod
    def to_string(self) -> str:
        ''''''

    @abstractmethod
    def save_results(self, results, save_dir) -> None:
         '''Method for saving simulation results.'''

class Basic_Eval(Abstract_Evaluator):
    def __init__(self, supervisor:np.ndarray, eval_steps:int, sample_rate:int,\
                 dt:float, net_size:int) -> None:
        ''''''
        self.sup = supervisor
        self.eval_steps = eval_steps
        self.sample_rate = sample_rate
        self.dt = dt
        self.N = net_size
        self.samples = self.eval_steps//self.sample_rate
        self.dim = self.sup.shape[1] if len(self.sup.shape) == 2 else 1

    def test_network(self, network: nn.Abstract_Network, current_step:int) -> Dict:
        ''''''
        state = network.state()
        test_error = 0
        z_test = np.zeros((self.samples, self.dim)).squeeze()

        for n in range(self.eval_steps):
            z = network.step(self.dt)[0]
            f = self.sup[(current_step + n) % self.sup.shape[0]]

            # Store testing data
            if (n % self.sample_rate) == 0:
                z_test[int(n/self.sample_rate)] = z

            test_error += (f-z) * (f-z)

        test_error = np.sqrt(np.sum(test_error) * self.dt) / self.eval_steps

        if self.dim != 1:
            test_error = np.mean(test_error)

        sum_entry = {'tst_err':test_error, 'phi':network.get_decoder(),\
                     'z_test':z_test, 'state':state} # Note: Can add more save items here

        return sum_entry

    def save_results(self, results, save_dir):
        '''Method for saving simulation results.'''

        results_df = pd.DataFrame(results)
        results_df.to_pickle(save_dir + 'sum_data.pkl')

    def to_string(self) -> str:
        return f'\n\tdt:{self.dt}'\
            + f'\n\teval_steps:{self.eval_steps}'\
            + f'\n\teval_step:{self.sample_rate}'