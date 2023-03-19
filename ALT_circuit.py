import pandas as pd
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import itertools

import pickle

from qulacs import QuantumState
from qulacs.state import inner_product
from qulacs import QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import DenseMatrix, PauliRotation
from qulacs.gate import to_matrix_gate, add

from qulacs.circuit import QuantumCircuitOptimizer
from qulacs import QuantumState, DensityMatrix
from qulacs.gate import Identity, X,Y,Z, Pauli
from qulacs.gate import H,S,Sdag, sqrtX,sqrtXdag,sqrtY,sqrtYdag
from qulacs.gate import T,Tdag
from qulacs.gate import RX,RY,RZ
from qulacs.gate import CNOT, CZ, SWAP
from qulacs import Observable, PauliOperator
import math
from qulacs.gate import DephasingNoise,DepolarizingNoise,TwoQubitDepolarizingNoise

###################### ALT XYZ rotation per layer ##################################
#noisy
def periodic_ALT_DP(num, depth, noise1Q_rates, noise2Q_rates, noiseRead_rates, params=None):
    '''
    constructs a noisy quantum circuit of an ALT ansatz similar to the one used in Ref. [1].
    Half of the qubits are entangled via CZ gates in a single layer.
    Entangling block is altered layer by layer.
    The model of the noise is a local depolarizing noise model considered in Ref. [1].
    Single-qubit depolarizing noise channels are inserted just before the measurement of each qubit as a model of the readout error.
    This function returns the circuit as a ParametricQuantumCircuit object of Qulacs [2].
    -------------------------------
    Reference
        [1] K. Ito, W. Mizukami, and K. Fujii, arXiv:2106.03390
        [2] Y. Suzuki et. al., arXiv:2011.13524
    -------------------------------
    Args:
        num (int) : number of qubits
        depth (int) : number of layers
        noise1Q_rates (list of float) : list of error probabilities of single-qubit errors
        noise2Q_rates (list of float) : list of error probabilities of two-qubit errors
        noiseRead_rates (list of float) : list of error probabilities of readout errors of each qubit
        params (list of float, optional) : list of initial parameters. If it is None, all parameters are initialized as 0.
    Returns:
        circuit (qulacs.ParametricQuantumCircuit)
    '''
    circuit = ParametricQuantumCircuit(num)
    para_count = 0
    noise1Q_count = 0
    noise2Q_count = 0
    noiseRead_count = 0
    if params is None:
        params = np.zeros(num * 3 * depth)
    ent_unit_size = num // 2
    unit_move_width = ent_unit_size // 2
    alternating_offset = 0
    ent_unit_counter = 0

    for i in range(depth):
        #single-qubit operations
        for j in range(num):
            circuit.add_parametric_RX_gate(j, params[para_count])
            para_count += 1
            circuit.add_gate(DepolarizingNoise(j, noise1Q_rates[noise1Q_count]))
            noise1Q_count += 1
            circuit.add_parametric_RY_gate(j, params[para_count])
            para_count += 1
            circuit.add_gate(DepolarizingNoise(j, noise1Q_rates[noise1Q_count]))
            noise1Q_count += 1
            circuit.add_parametric_RZ_gate(j, params[para_count])
            para_count += 1
            circuit.add_gate(DepolarizingNoise(j, noise1Q_rates[noise1Q_count]))
            noise1Q_count += 1
        #two-qubit operations
        jlist = (np.arange(num-1) + alternating_offset)%num
        for j in jlist:
            if (ent_unit_counter == (ent_unit_size - 1)):
                ent_unit_counter = 0
                continue
            circuit.add_CZ_gate(j, (j+1)%num)
            circuit.add_gate(TwoQubitDepolarizingNoise(j, (j+1)%num, noise2Q_rates[noise2Q_count]))
            noise2Q_count += 1
            ent_unit_counter += 1
        ent_unit_counter = 0
        alternating_offset = (alternating_offset + unit_move_width) % num
    #readout error
    for j in range(num):
        circuit.add_gate(DepolarizingNoise(j, noiseRead_rates[noiseRead_count]))
        noiseRead_count += 1    
    return circuit

#noiseless
def periodic_ALT_noiseless(num, depth, params=None, Dagger=False, Output_gate_count=False):
    '''
    constructs a noiseless quantum circuit of an ALT ansatz similar to the one used in Ref. [1].
    Half of the qubits are entangled via CZ gates in a single layer.
    Entangling block is altered layer by layer.
    This function returns the circuit as a ParametricQuantumCircuit object of Qulacs [2].
    -------------------------------
    Reference
        [1] K. Ito, W. Mizukami, and K. Fujii, arXiv:2106.03390
        [2] Y. Suzuki et. al., arXiv:2011.13524
    -------------------------------
    Args:
        num (int) : number of qubits
        depth (int) : number of layers
        params (list of float, optional) : list of initial parameters. If it is None, all parameters are initialized as 0.
        Dagger (bool, optional) : If it is True, the Hermitian conjugate of the circuit is returned.
        Output_gate_count (bool, optional) : If its value is True, the respective counts of 1-qubit gates and 2-qubit gates are additionally returned.
    Returns:
        circuit (qulacs.ParametricQuantumCircuit)
        gate_1Qcount (int) (only if Output_gate_count=True)
        gate_2Qcount (int) (only if Output_gate_count=True)
    '''
    circuit = ParametricQuantumCircuit(num)
    para_count = 0
    if params is None:
        params = np.zeros(num * 3 * depth)
    ent_unit_size = num // 2
    unit_move_width = ent_unit_size // 2
    alternating_offset = 0
    ent_unit_counter = 0
    gate_1Qcount = 0
    gate_2Qcount = 0
    #Dagger ----------------------
    if (Dagger):    
        params_rev = np.flip(params)
        alternating_offset = (alternating_offset + unit_move_width*(depth - 1))%num
        for i in range(depth):
            jlist = (np.arange(num-1) + alternating_offset)%num
            for j in jlist:
                if (ent_unit_counter == (ent_unit_size - 1)):
                    ent_unit_counter = 0
                    continue
                circuit.add_CZ_gate(j, (j+1)%num)
                gate_2Qcount += 1
                ent_unit_counter += 1
            for j in range(num):
                circuit.add_parametric_RZ_gate(num - j - 1, - params_rev[para_count])
                gate_1Qcount += 1
                para_count += 1
                circuit.add_parametric_RY_gate(num - j - 1, - params_rev[para_count])
                gate_1Qcount += 1
                para_count += 1
                circuit.add_parametric_RX_gate(num - j - 1, - params_rev[para_count])
                gate_1Qcount += 1
                para_count += 1
            ent_unit_counter = 0
            alternating_offset = (alternating_offset - unit_move_width)%num
        if Output_gate_count:
            return circuit, gate_1Qcount, gate_2Qcount
        else:
            return circuit
    #----------------- End Dagger
    for i in range(depth):
        for j in range(num):
            circuit.add_parametric_RX_gate(j, params[para_count])
            gate_1Qcount += 1
            para_count += 1
            circuit.add_parametric_RY_gate(j, params[para_count])
            gate_1Qcount += 1
            para_count += 1
            circuit.add_parametric_RZ_gate(j, params[para_count])
            gate_1Qcount += 1
            para_count += 1
        jlist = (np.arange(num-1) + alternating_offset)%num
        for j in jlist:
            if (ent_unit_counter == (ent_unit_size - 1)):
                ent_unit_counter = 0
                continue
            circuit.add_CZ_gate(j, (j+1)%num)
            gate_2Qcount += 1
            ent_unit_counter += 1
        ent_unit_counter = 0
        alternating_offset = (alternating_offset + unit_move_width) % num
    if Output_gate_count:
        return circuit, gate_1Qcount, gate_2Qcount
    else:
        return circuit
    
###################
#QGT diagonal
def G_i_ALT_VPofDP(num, depth, params):
    '''
    calculates G_i(params) of ALT for virtual parameters associated with the local depolarizing noise model
    Each G_i is 4 times each diagonal component of the Fubini-Study metric.
    See the following reference for detail.
    -------------------------------
    Reference [1] arXiv:2106.03390
    -------------------------------
    Args:
        num (int) : number of qubits
        depth (int) : number of layers
        params (list of float) : list of parameters at which G_i is calculated
    Returns:
        G_diag_1Q (list of float) : G_i for virtual parameters associated with 1-qubit depolarizing channels including readout error
        G_diag_2Q (list of float) : G_i for virtual parameters associated with 2-qubit depolarizing channels
    '''
    state = QuantumState(num)
    state.set_zero_state()
    circuit = ParametricQuantumCircuit(num)
    para_count = 0
    if params is None:
        params = np.zeros(num * 3 * depth)
    ent_unit_size = num // 2
    unit_move_width = ent_unit_size // 2
    alternating_offset = 0
    ent_unit_counter = 0
    G_diag_1Q = []
    G_diag_2Q = []
    pauli1Q_str = ["X", "Y", "Z"]
    pauli_str = ["I", "X", "Y", "Z"]
    for i in range(depth):
        for j in range(num):
            #single-qubit gates
            circuit.add_parametric_RX_gate(j, params[para_count])
            state.set_zero_state()
            circuit.update_quantum_state(state)
            for ps in pauli1Q_str:
                Obs = PauliOperator(ps + " " + str(j), 1.)
                Gd = Obs.get_expectation_value(state) ** 2
                G_diag_1Q.append(Gd)
            para_count += 1
            circuit.add_parametric_RY_gate(j, params[para_count])
            state.set_zero_state()
            circuit.update_quantum_state(state)
            for ps in pauli1Q_str:
                Obs = PauliOperator(ps + " " + str(j), 1.)
                Gd = Obs.get_expectation_value(state) ** 2
                G_diag_1Q.append(Gd)
            para_count += 1
            circuit.add_parametric_RZ_gate(j, params[para_count])
            state.set_zero_state()
            circuit.update_quantum_state(state)
            for ps in pauli1Q_str:
                Obs = PauliOperator(ps + " " + str(j), 1.)
                Gd = Obs.get_expectation_value(state) ** 2
                G_diag_1Q.append(Gd)
            para_count += 1
        #two-qubit gates
        jlist = (np.arange(num-1) + alternating_offset)%num
        for j in jlist:
            if (ent_unit_counter == (ent_unit_size - 1)):
                ent_unit_counter = 0
                continue
            circuit.add_CZ_gate(j, (j+1)%num)
            state.set_zero_state()
            circuit.update_quantum_state(state)
            for ps1 in pauli_str:
                for ps2 in pauli_str:
                    if (ps1+ps2)=="II":
                        continue
                    Obs = PauliOperator(ps1 + " " + str(j) + " " + ps2 + " " + str((j+1)%num), 1.)
                    Gd = Obs.get_expectation_value(state) ** 2
                    G_diag_2Q.append(Gd)
            ent_unit_counter += 1
        ent_unit_counter = 0
        alternating_offset = (alternating_offset + unit_move_width) % num
    #readout error
    for j in range(num):
        for ps in pauli1Q_str:
            Obs = PauliOperator(ps + " " + str(j), 1.)
            Gd = Obs.get_expectation_value(state) ** 2
            G_diag_1Q.append(Gd)
    G_diag_1Q = np.array(G_diag_1Q)
    G_diag_2Q = np.array(G_diag_2Q)
    G_diag_1Q = 1. - G_diag_1Q
    G_diag_2Q = 1. - G_diag_2Q
    return G_diag_1Q.real, G_diag_2Q.real

