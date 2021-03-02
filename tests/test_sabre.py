# -*- coding : utf-8 -*-
"""
@authors Quentin Delamea <quentin.delamea@atos.net>
@intern
@copyright 2020-2019  Bull S.A.S.  -  All rights reserved.
@file qat-plugins/tests/test_sabre.py
@brief Tests Sabre plugin
@namespace qat-plugins.tests.test_sabre

    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
"""

import random as rd
from typing import List

import networkx as nx
import numpy as np
import pytest

from qat.comm.exceptions.ttypes import PluginException
from qat.core import Topology, TopologyType, Result, Batch, HardwareSpecs, Circuit, Observable, Term
from qat.core.quameleon import QuameleonPlugin
from qat.lang.AQASM import Program, H, X, Y, Z, CNOT, RX, RY, RZ, PH
from qat.lang.AQASM.qftarith import QFT
from qat.plugins import Sabre
from qat.pylinalg import PyLinalg

# Be careful the simulation is exponentially time-consuming when the number of qubits increase
min_nbqbit = 2
max_nbqbit = 10


def check_same_states(result_1: Result, result_2: Result) -> None:
    """
    Given two results from two simulations, checks if the two results have the same states.

    Args:
        result_1 (Result): result of the first simulation
        result_2 (Result): result of the second simulation

    Return:
        Nothing
    """

    # Get all state of result_1
    states_result_1 = []
    for sample in result_1:
        states_result_1.append(sample.state.int)

    # Get all state of result_2
    states_result_2 = []
    for sample in result_2:
        states_result_2.append(sample.state.int)

    states_result_1.sort()
    states_result_2.sort()

    # Check content equality
    assert states_result_1 == states_result_2


def check_same_probabilities(result_1: Result, result_2: Result) -> None:
    """
    Given two results from two simulations, checks if the two results have the same states.

    Args:
        result_1 (Result): result of the first simulation
        result_2 (Result): result of the second simulation

    Return:
        Nothing
    """

    # Get all probabilities of result_1
    prob_result_1 = []
    for sample in result_1:
        prob_result_1.append(sample.probability)

    # Get all probabilities of result_2
    prob_result_2 = []
    for sample in result_2:
        prob_result_2.append(sample.probability)

    prob_result_1.sort()
    prob_result_2.sort()

    # Check content equality
    assert prob_result_1 == pytest.approx(prob_result_2)


def check_same_state_properties(result_1: Result, result_2: Result, amplitudes: bool = True) -> None:
    """
    Given two results of two different simulation and assuming the two result have the same states (see above),
    checks if states of the two simulations have the same amplitudes and probabilities.

    Args:
        result_1 (Result): result of the first simulation
        result_2 (Result): result of the second simulation
        amplitudes (bool, optional): set to True to check equality of amplitudes for each state, False otherwise, default True

    Return:
        Nothing
    """

    # Get properties of states for result_1
    states_1 = {}
    for sample in result_1:
        states_1[sample.state.int] = {'probability': sample.probability, 'amplitude': sample.amplitude}

    # Get properties of states for result_2
    states_2 = {}
    for sample in result_2:
        states_2[sample.state.int] = {'probability': sample.probability, 'amplitude': sample.amplitude}

    # Iterate over states and check the almost equality of properties
    for state in states_1.keys():
        if amplitudes:
            assert np.linalg.norm(states_1[state]['amplitude']) == pytest.approx(np.linalg.norm(states_2[state]['amplitude']))
            assert np.angle(states_1[state]['amplitude']) == pytest.approx(np.angle(states_2[state]['amplitude']))
        assert states_1[state]['probability'] == pytest.approx(states_2[state]['probability'])


def check_results_equality(result_1: Result, result_2: Result, amplitude: bool = True) -> None:
    """
    Checks the result of two simulations are "equal" taking approximation into account.

    Args:
        result_1 (Result): reference result
        result_2 (Result): tested result
        amplitude (bool, optional): set to True to check equality of amplitudes, False otherwise, default True

    Returns:
        Nothing
    """

    # First we check the two results have the same states
    check_same_states(result_1, result_2)

    # Then we check the two results have the same probabilities
    check_same_probabilities(result_1, result_2)

    # Finally we check the two results have the same stats properties
    if amplitude:
        check_same_state_properties(result_1, result_2)
    else:
        check_same_state_properties(result_1, result_2, amplitude)


def check_measures_equality(measure_1: Result, measure_2: Result) -> None:
    """
    Checks equality of mean value measured for simulations using observable.

    Args:
        measure_1 (Result): reference measure
        measure_2 (Result): tested measure

    Returns:
        Nothing
    """

    assert measure_1.value is not None
    assert measure_2.value is not None
    assert measure_1.value == pytest.approx(measure_2.value)


def check_circuits_equality(circuit_1: Circuit, circuit_2: Circuit) -> None:
    """
    Circuit object doesn't provide equality override, so this function checks equality of two quantum circuits.

    Args:
        circuit_1 (Circuit): reference quantum circuit
        circuit_2 (Circuit): tested quantum circuit

    Returns:
        Nothing
    """

    # We check the equality of circuits properties one by one excepted the qregister which can not be compared
    assert circuit_1.ops == circuit_2.ops
    assert circuit_1.gateDic == circuit_2.gateDic
    assert circuit_1.var_dic == circuit_2.var_dic


def generate_custom_topologies(nbqbit: int) -> List[Topology]:
    """
    Generates several custom topologies for a given number of qubits.

    Args:
        nbqbit (int): number of qubits in the topology

    Returns:
        List<Topology>: a list containing the topologies
    """

    # We use NetworkX graph generator and the Topology class method from_nx to build topologies
    topology_list = [
        Topology.from_nx(nx.path_graph(nbqbit)),
        Topology.from_nx(nx.complete_graph(nbqbit)),
        Topology.from_nx(nx.star_graph(nbqbit - 1)),
        Topology.from_nx(nx.wheel_graph(nbqbit)),
        Topology.from_nx(nx.cycle_graph(nbqbit))
    ]
    return topology_list


def generate_qft_circuit(nbqbits: int, inline: bool = True) -> Circuit:
    """
    Creates a  quantum circuit composed of an initialization and a QFT applied on all qubits.

    Args:
          nbqbits (int): number of qubits in the circuit
          inline (bool, optional): True to inline the circuit, False otherwise, default True

    Returns:
          Circuit: a quantum circuit containing random gates
    """

    # Initialize program and qregister
    prog = Program()
    qbits = prog.qalloc(nbqbits)

    # Qubits initialization (to have a non |0^n> state)
    for qbit in qbits:
        prog.apply(H, qbit)
        prog.apply(Z, qbit)

    # Apply QFT on all qubits
    prog.apply(QFT(nbqbits), qbits)

    return prog.to_circ(inline=inline)


def generate_random_circuit(nbqbits: int) -> Circuit:
    """
    Creates a random quantum circuit composed of one or two qubits gates.

    Args:
          nbqbits (int): number of qubits in the circuit

    Returns:
          Circuit: quantum circuit
    """

    # Build quantum program and quantum register
    prog = Program()
    qbits = prog.qalloc(nbqbits)

    # Build the gate set
    gate_set = [H, X, Y, Z, CNOT, RX, RY, RZ, PH]

    # Determine randomly the number of gates applied on the program
    nb_gates = rd.randint(10 * nbqbits // 2, 10 * nbqbits)

    # Extract a random list of gates
    gates = [rd.choice(gate_set) for _ in range(nb_gates)]

    # Apply gates on prog
    for gate in gates:
        if gate in [RX, RY, RZ, PH]:  # One qubit parametrised gate
            # Get a random angle
            angle = rd.random() * 2 * np.pi
            # Determine randomly if the gate is controlled
            is_ctrl = rd.randint(0, 1)
            if is_ctrl == 0:
                prog.apply(gate(angle), qbits[rd.randint(0, nbqbits - 1)])
            elif is_ctrl == 1:
                qbit_1, qbit_2 = rd.sample(list(range(nbqbits)), 2)
                prog.apply(gate(angle).ctrl(), [qbits[qbit_1], qbits[qbit_2]])
        elif gate in [H, X, Y, Z]:  # Two qubits non-parametrised gate
            # Determine randomly if the gate is controlled
            is_ctrl = rd.randint(0, 1)
            if is_ctrl == 0:
                prog.apply(gate, qbits[rd.randint(0, nbqbits - 1)])
            elif is_ctrl == 1:
                qbit_1, qbit_2 = rd.sample(list(range(nbqbits)), 2)
                prog.apply(gate.ctrl(), [qbits[qbit_1], qbits[qbit_2]])
        else:  # Two qubits gate
            qbit_1, qbit_2 = rd.sample(list(range(nbqbits)), 2)
            prog.apply(gate, [qbits[qbit_1], qbits[qbit_2]])

    return prog.to_circ()


def generate_random_observable(nbqbit: int) -> Observable:
    """
    Generate a random observable.

    Args:
        nbqbit (int): number of qubits

    Returns:
        Observable: an observable
    """

    # Determine randomly the number of Pauli terms in the observable
    nb_terms = rd.randint(1, nbqbit)

    pauli_terms = []
    for _ in range(nb_terms):
        # Determine randomly the number of qubits involve in the Pauli term
        nb_qbit_term = rd.randint(1, nbqbit)
        # Build randomly the tensor of the term
        tensor = ''.join([rd.choice(['X', 'Y', 'Z']) for _ in range(nb_qbit_term)])
        # Create and add the term to the list
        pauli_terms.append(Term(rd.random(), tensor, rd.sample(range(nbqbit), nb_qbit_term)))
    # build and return the observable
    return Observable(nbqbit, pauli_terms=pauli_terms, constant_coeff=rd.random())


class TestSabreExceptions:
    """
    Class containing all the test functions of Sabre PluginException.
    """

    def test_unknown_topology_type(self) -> None:
        """
        Checks Sabre raises PluginException for an unknown topology type.
        """
        with pytest.raises(PluginException):
            circuit = generate_qft_circuit(max_nbqbit)
            qpu = Sabre() | (QuameleonPlugin(topology=Topology(type=-5)) | PyLinalg())
            qpu.submit(circuit.to_job())

    def test_custom_topology_without_graph(self) -> None:
        """
        Checks Sabre raises PluginException if TopologyType is CUSTOM but no graph is provided.
        """
        with pytest.raises(PluginException):
            circuit = generate_qft_circuit(max_nbqbit)
            qpu = Sabre() | (QuameleonPlugin(topology=Topology(type=TopologyType.CUSTOM)) | PyLinalg())
            qpu.submit(circuit.to_job())

    def test_gate_more_two_qubits(self) -> None:
        """
        Checks Sabre raises PluginException if there is a circuit containing a gate with more than two qubits.
        """
        with pytest.raises(PluginException):
            circuit = generate_qft_circuit(max_nbqbit, inline=False)
            qpu = Sabre() | (QuameleonPlugin(topology=Topology(type=TopologyType.LNN)) | PyLinalg())
            qpu.submit(circuit.to_job())

    def test_too_much_qubits(self) -> None:
        """
        Checks Sabre raises PluginException if there are more qubits in the circuit than in the topology.
        """
        with pytest.raises(PluginException):
            circuit = generate_qft_circuit(max_nbqbit)
            qpu = Sabre() | (QuameleonPlugin(topology=Topology(type=TopologyType.CUSTOM, graph={0: [1], 1: [0]})) | PyLinalg())
            qpu.submit(circuit.to_job())


class TestSabreOnQFT:
    """
    Class containing functions which test if Sabre plugin works well for QFT circuit on different topologies.
    """

    def test_qft_all_to_all_topology(self) -> None:
        """
        Checks that Sabre doesn't modify the circuit if the topology is of type ALL_TO_ALL.
        """

        for nbqbit in range(min_nbqbit, max_nbqbit):
            circuit = generate_qft_circuit(nbqbit)

            sabre = Sabre()
            batch = Batch(jobs=[circuit.to_job()])
            hardware_specs = HardwareSpecs()
            batch_result = sabre.compile(batch, hardware_specs)
            computed_circuit = batch_result.jobs[0].circuit

            check_circuits_equality(circuit, computed_circuit)

    def test_qft_lnn(self) -> None:
        """
        Tests Sabre on a QFT for LNN topologies for different number of qubits.
        """

        for nbqbit in range(3, max_nbqbit):
            circuit = generate_qft_circuit(nbqbit)

            qpu_1 = PyLinalg()
            result_1 = qpu_1.submit(circuit.to_job())

            qpu_2 = Sabre() | (QuameleonPlugin(topology=Topology(type=TopologyType.LNN)) | PyLinalg())
            result_2 = qpu_2.submit(circuit.to_job())

            check_results_equality(result_1, result_2)

    def test_qft_custom_usual_topologies(self) -> None:
        """
        Tests Sabre on a QFT for some usual custom topologies for different number of qubits.
        """

        for nbqbit in range(5, max_nbqbit):
            circuit = generate_qft_circuit(nbqbit)

            qpu_1 = PyLinalg()
            result_1 = qpu_1.submit(circuit.to_job())

            for topology in generate_custom_topologies(nbqbit):
                print('Number of qubits : ', nbqbit)
                print('Current topology : ', topology)
                qpu_2 = Sabre() | (QuameleonPlugin(topology=topology) | PyLinalg())
                result_2 = qpu_2.submit(circuit.to_job())
                check_results_equality(result_1, result_2)


class TestSabreForSamplingMode:
    """
    Class containing functions which test if Sabre plugin works well on common circuit for different classical
    topologies when all qubits are measured at the end of the simulation.
    """

    def test_lnn_topology(self) -> None:
        """
        Tests Sabre on common circuit for LNN topologies for different number of qubits.
        """

        for nbqbit in range(2 * min_nbqbit, max_nbqbit):
            circuit = generate_random_circuit(rd.randint(nbqbit // 2, nbqbit))

            qpu_1 = PyLinalg()
            result_1 = qpu_1.submit(circuit.to_job())

            qpu_2 = Sabre() | (QuameleonPlugin(topology=Topology(type=TopologyType.LNN)) | PyLinalg())
            result_2 = qpu_2.submit(circuit.to_job())

            check_results_equality(result_1, result_2)

    def test_custom_usual_topologies(self) -> None:
        """
        Tests Sabre on common circuit for some usual custom topologies for different number of qubits.
        """

        for nbqbit in range(2 * min_nbqbit, max_nbqbit):
            circuit = generate_random_circuit(rd.randint(nbqbit // 2, nbqbit))

            qpu_1 = PyLinalg()
            result_1 = qpu_1.submit(circuit.to_job())

            for topology in generate_custom_topologies(nbqbit):
                qpu_2 = Sabre() | (QuameleonPlugin(topology=topology) | PyLinalg())
                result_2 = qpu_2.submit(circuit.to_job())

                check_results_equality(result_1, result_2, amplitude=False)

            if nbqbit % 2 == 0:
                topology = Topology.from_nx(nx.grid_2d_graph(nbqbit // 2, nbqbit // 2))
                qpu_2 = Sabre() | (QuameleonPlugin(topology=topology) | PyLinalg())
                result_2 = qpu_2.submit(circuit.to_job())

                check_results_equality(result_1, result_2, amplitude=False)


class TestSabreForPartialMeasuringMode:
    """
    Class containing functions which test if Sabre plugin works well on common circuit for different classical
    topologies when a part of the qubits are measured at the end of the simulation.
    """

    def test_lnn_topology(self) -> None:
        """
        Tests Sabre on common circuit for LNN topologies for different number of qubits.
        """

        for nbqbit in range(2 * min_nbqbit, max_nbqbit):
            nbqbit_circuit = rd.randint(nbqbit // 2, nbqbit)
            circuit = generate_random_circuit(nbqbit_circuit)

            nb_measured_qubits = rd.randint(1, nbqbit_circuit)
            measured_qubits = rd.sample(range(nbqbit_circuit), nb_measured_qubits)
            measured_qubits.sort()

            qpu_1 = PyLinalg()
            result_1 = qpu_1.submit(circuit.to_job(qubits=measured_qubits))

            qpu_2 = Sabre() | (QuameleonPlugin(topology=Topology(type=TopologyType.LNN)) | PyLinalg())
            result_2 = qpu_2.submit(circuit.to_job(qubits=measured_qubits))

            check_results_equality(result_1, result_2, amplitude=False)

    def test_custom_usual_topologies(self) -> None:
        """
        Tests Sabre on common circuit for some usual custom topologies for different number of qubits.
        """

        for nbqbit in range(2 * min_nbqbit, max_nbqbit):
            nbqbit_circuit = rd.randint(nbqbit // 2, nbqbit)
            circuit = generate_random_circuit(nbqbit_circuit)

            nb_measured_qubits = rd.randint(1, nbqbit_circuit)
            measured_qubits = rd.sample(range(nbqbit_circuit), nb_measured_qubits)
            measured_qubits.sort()

            qpu_1 = PyLinalg()
            result_1 = qpu_1.submit(circuit.to_job(qubits=measured_qubits))

            for topology in generate_custom_topologies(nbqbit):
                qpu_2 = Sabre() | (QuameleonPlugin(topology=topology) | PyLinalg())
                result_2 = qpu_2.submit(circuit.to_job(qubits=measured_qubits))

                check_results_equality(result_1, result_2, amplitude=False)

            if nbqbit % 2 == 0:
                topology = Topology.from_nx(nx.grid_2d_graph(nbqbit // 2, nbqbit // 2))
                qpu_2 = Sabre() | (QuameleonPlugin(topology=topology) | PyLinalg())
                result_2 = qpu_2.submit(circuit.to_job(qubits=measured_qubits))

                check_results_equality(result_1, result_2, amplitude=False)


class TestSabreForObservable:
    """
    Class containing functions which test if Sabre plugin works well on common circuit for different classical
    topologies when an observable is measured at the end of the simulation.
    """

    def test_lnn_topology(self) -> None:
        """
        Tests Sabre on common circuit for LNN topologies for different number of qubits.
        """

        for nbqbit in range(2 * min_nbqbit, max_nbqbit):
            nbqbit_circuit = rd.randint(nbqbit // 2, nbqbit)
            circuit = generate_random_circuit(nbqbit_circuit)

            observable = generate_random_observable(nbqbit_circuit)

            qpu_1 = PyLinalg()
            measure_1 = qpu_1.submit(circuit.to_job("OBS", observable=observable))

            qpu_2 = Sabre() | (QuameleonPlugin(topology=Topology(type=TopologyType.LNN)) | PyLinalg())
            measure_2 = qpu_2.submit(circuit.to_job("OBS", observable=observable))

            check_measures_equality(measure_1, measure_2)

    def test_custom_usual_topologies(self) -> None:
        """
        Tests Sabre on common circuit for some usual custom topologies for different number of qubits.
        """

        for nbqbit in range(2 * min_nbqbit, max_nbqbit):
            nbqbit_circuit = rd.randint(nbqbit // 2, nbqbit)
            circuit = generate_random_circuit(nbqbit_circuit)

            observable = generate_random_observable(nbqbit_circuit)

            qpu_1 = PyLinalg()
            measure_1 = qpu_1.submit(circuit.to_job("OBS", observable=observable, nbshots=5))

            for topology in generate_custom_topologies(nbqbit):
                qpu_2 = Sabre() | (QuameleonPlugin(topology=topology) | PyLinalg())
                measure_2 = qpu_2.submit(circuit.to_job("OBS", observable=observable, nbshots=5))

                check_measures_equality(measure_1, measure_2)

            if nbqbit % 2 == 0:
                topology = Topology.from_nx(nx.grid_2d_graph(nbqbit // 2, nbqbit // 2))
                qpu_2 = Sabre() | (QuameleonPlugin(topology=topology) | PyLinalg())
                measure_2 = qpu_2.submit(circuit.to_job("OBS", observable=observable, nbshots=5))

                check_measures_equality(measure_1, measure_2)


class TestParticularCases:
    """
    Class containing functions which test if Sabre plugin works well for some particular cases.
    """

    def test_all_to_all_topology(self) -> None:
        """
        Checks that Sabre doesn't modify the circuit if the topology is of type ALL_TO_ALL.
        """

        for nbqbit in range(min_nbqbit, max_nbqbit):
            circuit = generate_random_circuit(nbqbit)

            sabre = Sabre()
            batch = Batch(jobs=[circuit.to_job()])
            hardware_specs = HardwareSpecs()
            batch_result = sabre.compile(batch, hardware_specs)
            computed_circuit = batch_result.jobs[0].circuit

            check_circuits_equality(circuit, computed_circuit)

    def test_already_executable_circuit(self) -> None:
        """
        Tests Sabre on a fully executable circuit which means all gates are applied on qubits which are connected on the
        hardware.
        """

        for nbqbit in range(min_nbqbit, max_nbqbit):
            prog = Program()
            qbits = prog.qalloc(nbqbit)

            for i in range(len(qbits) - 1):
                prog.apply(H, qbits[i])
                prog.apply(Z, qbits[i])
                prog.apply(X.ctrl(), qbits[i + 1], qbits[i])

            circuit = prog.to_circ(inline=True)

            sabre = Sabre()
            batch = Batch(jobs=[circuit.to_job()])
            hardware_specs = HardwareSpecs()
            batch_result = sabre.compile(batch, hardware_specs)
            computed_circuit = batch_result.jobs[0].circuit

            check_circuits_equality(circuit, computed_circuit)

    def test_no_state_modification_circuit(self) -> None:
        """
        We apply Sabre on a circuit which doesn't modify the initial state (|0^n> here) and we verify Sabre circuit
        modifications don't modify the state.
        """

        for nbqbit in range(min_nbqbit, max_nbqbit):
            prog = Program()
            qbits = prog.qalloc(nbqbit)

            random_angles = [rd.random() * 2 * np.pi for _ in range(3 * nbqbit)]

            for i in range(len(qbits)):
                prog.apply(RX(random_angles[3 * i]), qbits[i])
                prog.apply(RX(random_angles[3 * i + 1]), qbits[i])
                prog.apply(RX(random_angles[3 * i + 2]), qbits[i])

            prog.apply(QFT(nbqbit), qbits)
            prog.apply(QFT(nbqbit).dag(), qbits)

            for i in range(len(qbits)):
                prog.apply(RX(random_angles[3 * i]).dag(), qbits[i])
                prog.apply(RX(random_angles[3 * i + 1]).dag(), qbits[i])
                prog.apply(RX(random_angles[3 * i + 2]).dag(), qbits[i])

            circuit = prog.to_circ(inline=True)

            for topology in generate_custom_topologies(nbqbit):
                qpu = Sabre() | (QuameleonPlugin(topology=topology) | PyLinalg())
                result = qpu.submit(circuit.to_job())
                assert result.raw_data[0].state.int == 0
