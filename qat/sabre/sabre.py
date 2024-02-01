# -*- coding : utf-8 -*-
"""
@authors Quentin Delamea <quentin.delamea@atos.net>
@intern
@copyright 2020  Bull S.A.S.  -  All rights reserved.
@file qat/sabre/sabre.py
@brief Simple implementation of Sabre algorithm
@namespace qat.sabre
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


SOME IMPORTANT POINTS ABOUT SABRE ALGORITHM:
    * It works for most of regular topologies (common ones) but sometimes for singular topologies the algorithm might
    not converge;
    * It works only with compliment topologies which means there are enough qubits on the chip and enough links too;
    * It only deals with one and two qubits gates.

SOME IMPORTANT POINTS ABOUT THIS IMPLEMENTATION:
    * When you apply this plugin you must inline your circuit;
    * The topology must be a bidirectionnal topology;
    * Intermediate measures are not supported by this implementation;
    * If the number of qubits of the hardware is not specified then the number is set to number of qubits of the
    topology.
"""

# Standard modules imports
import copy
import random as rd
from typing import Generator, Tuple, List

# External modules imports
import networkx as nx

# myQLM modules imports
from qat.comm.datamodel.ttypes import Op, OpType
from qat.comm.exceptions.ttypes import PluginException, ErrorType
from qat.comm.shared.ttypes import ProcessingType
from qat.core import Batch, Job, HardwareSpecs, TopologyType, Circuit
from qat.core.plugins import AbstractPlugin


def iterate_job_and_topology(batch: Batch, hardware_specs: HardwareSpecs) \
        -> Generator[Tuple[Job, nx.Graph, int], None, None]:
    """
    Iterates over jobs, chip coupling graph and the number of qubits on the chip.

    Args:
        batch (Batch): computing batch
        hardware_specs (HardwareSpecs): chip hardware specs

    Returns:
        Iterator<Job, networkx.Graph, int>: iterator over jobs, chip coupling graph and qubits number
    """

    # Case All to All topology has been already checked at this stage
    # We check the other cases
    if hardware_specs.topology.type == TopologyType.LNN:
        for job in batch.jobs:
            # Get the number of qubits
            circuit = job.circuit
            nbqbits = circuit.nbqbits
            # Build chip coupling graph
            chip_coupling_graph = nx.path_graph(nbqbits)
            # Build the iteration
            yield job, chip_coupling_graph, nbqbits
    elif hardware_specs.topology.type == TopologyType.CUSTOM:
        # First if there is no specified graph we raise an error
        if hardware_specs.topology.graph is None:
            raise PluginException(
                message='TopologyType is CUSTOM but no topology graph is given.',
                code=ErrorType.ABORT,
                modulename="qat.plugins",
                file="qat/sabre/sabre.py"
            )
        # We build graph associated to custom topology
        chip_coupling_graph = nx.Graph(hardware_specs.topology.graph)

        # Get number of qubits
        if hardware_specs.nbqbits is None:
            nbqbits = len(hardware_specs.topology.graph.keys())
        else:
            nbqbits = hardware_specs.nbqbits

        for job in batch.jobs:
            # If there is less physical qubits than logical ones we raise an exception
            if nbqbits < job.circuit.nbqbits:
                raise PluginException(
                    message='Circuit requires more qubits than provided in the topology',
                    code=ErrorType.ABORT,
                    modulename="qat.plugins",
                    file="qat/sabre/sabre.py"
                )
            # Build the iteration
            yield job, chip_coupling_graph, nbqbits
    else:
        # The algorithm can only deal with All_TO_All, LNN and CUSTOM topologies
        raise PluginException(
            message='TopologyType doesn\'t correspond to any supported type.',
            code=ErrorType.ABORT,
            modulename="qat.plugins",
            file="qat/sabre/sabre.py"
        )


def circuit_to_dag(circuit: Circuit, nbqbits: int) -> nx.DiGraph:
    """
    Converts a quantum circuit into Directed Acyclic Graph where each node represents a two qubits gate and all
    single qubit gates executed on these two qubits before following two qubits gate apply on one of these
    qubits. The edges connect the gate groups in the order in which they are executed.

    Args:
        circuit (Circuit): quantum circuit
        nbqbits (int): the number of qubits on the chip

    Returns:
        networkx.DiGraph: Directed Acyclic Graph representing the quantum circuit.
    """

    # Initiate current layer
    current_layer = ['BEG'] * nbqbits

    # Initiate the DAG with a BEG node
    circuit_dag = nx.DiGraph()
    circuit_dag.add_node('BEG', ops=[], exec=True)

    # Initiate group index
    group_index = 0

    for op in circuit.ops:
        if len(op.qbits) > 2:
            # The algorithm can only deal with one or two qubits gates
            raise PluginException(
                message='Sabre only support one or two qubits gates.',
                code=ErrorType.ABORT,
                modulename="qat.plugins",
                file="qat/sabre/sabre.py"
            )
        if len(op.qbits) == 1:
            # Add the gate to corresponding group
            group_name = current_layer[op.qbits[0]]
            circuit_dag.nodes[group_name]['ops'].append(op)
        elif len(op.qbits) == 2:
            # Update group name
            group_index += 1
            group_name = 'g' + str(group_index)

            # Add node for the new group
            circuit_dag.add_node(group_name, ops=[op], exec=False)

            # Link new group with its dependencies
            circuit_dag.add_edges_from([
                (current_layer[op.qbits[0]], group_name),
                (current_layer[op.qbits[1]], group_name)
            ])

            # Update front layer
            current_layer[op.qbits[0]] = group_name
            current_layer[op.qbits[1]] = group_name
        else:
            # if none of the above conditions are verified we raise an exception
            raise PluginException(
                message='Encountered an none supported gate.',
                code=ErrorType.ABORT,
                modulename="qat.plugins",
                file="qat/sabre/sabre.py"
            )
    return circuit_dag


class Mapping:
    """
    Class describing invertible mapping from physical qubits on quantum device to logical qubits in quantum circuit.

    Attributes:
        indexed_by_physical (dic<int, int>): logical qubit indexed by its linked physical qubit
        indexed_by_logical (dic<int, int>): physical qubit indexed by its linked logical qubit
    """

    def __init__(self, nbqbits: int) -> None:
        """
        Initiates mapping as a trivial mapping where i-th physical qubit is linked to i-th logical qubit.
        If the number of logical qubits is less than the number of physical qubits we add as many logical qubits as
        necessary to obtain an equivalent number of logical and physical qubits.

        Args:
            nbqbits (int): the number of physical qubits on the chip

        Returns:
            None: nothing
        """

        self.indexed_by_physical = {}
        self.indexed_by_logical = {}

        # Here we initiate the  mapping
        for i in range(nbqbits):
            self.indexed_by_physical[i] = i
            self.indexed_by_logical[i] = i

    def get_by_physical_index(self, index: int) -> int:
        """
        Given a physical qubit, returns the logical corresponding qubit.

        Args:
            index (int): the physical qubit

        Returns:
            int: the linked logical qubit
        """

        return self.indexed_by_physical[index]

    def get_by_logical_index(self, index: int) -> int:
        """
        Given a logical qubit, returns the physical corresponding qubit.

        Args:
            index (int): the logical qubit

        Returns:
            int: the linked physical qubit
        """

        return self.indexed_by_logical[index]

    def update(self, swap: Tuple[int, int]) -> None:
        """
        Given a swap qubit, updates the mapping i.e switch physical and logical qubits between two pairs.

        Args:
            swap (Tuple<int,int>): pair of logical qubits

        Returns:
            None: nothing
        """

        # Extract logical qubits index from the swap
        logical_qbit_1, logical_qbit_2 = swap

        # Retrieve corresponding physical qubits
        physical_qbit_1 = self.indexed_by_logical[logical_qbit_1]
        physical_qbit_2 = self.indexed_by_logical[logical_qbit_2]

        # Update the mapping (i.e both dictionaries) by switching qubits
        self.indexed_by_logical[logical_qbit_1] = physical_qbit_2
        self.indexed_by_logical[logical_qbit_2] = physical_qbit_1
        self.indexed_by_physical[physical_qbit_1] = logical_qbit_2
        self.indexed_by_physical[physical_qbit_2] = logical_qbit_1


def is_executable(node: str, chip_coupling_graph: nx.Graph, mapping: Mapping, circuit_dag: nx.DiGraph) \
        -> bool:
    """
    Determines if a gate between two given logical qubits is executable on the hardware or not.

    Args:
        node (str): a node of the DAG
        chip_coupling_graph (networkx.Graph): the coupling graph of the chip
        mapping (Mapping): the mapping
        circuit_dag (networkx.DiGraph): quantum circuit corresponding DAG

    Returns:
        bool: True if the gate is executable on the hardware), False otherwise
    """

    # Extract logical qubits index from the swap
    logical_qbit_1, logical_qbit_2 = circuit_dag.nodes[node]['ops'][0].qbits

    # Retrieve corresponding physical qubits
    physical_qbit_1 = mapping.get_by_logical_index(logical_qbit_1)
    physical_qbit_2 = mapping.get_by_logical_index(logical_qbit_2)

    return chip_coupling_graph.has_edge(physical_qbit_1, physical_qbit_2)


def get_addable_successors(node: str, circuit_dag: nx.DiGraph) -> List[str]:
    """
    Finds successors of a gate which are addable to front layer (every dependencies have been already executed).

    Args:
        node (str): a node of the graph
        circuit_dag (networkx.DiGraph): the DAG representing the quantum circuit

    Return:
        List<str>: a list of all successor gates which have all their dependencies executed
    """

    # Initiate list of executable successors
    addable_successors = []

    # Iterate on all successor
    for successor in circuit_dag.successors(node):
        # While no non executed predecessor is find all_executed is set to True
        all_executed = True
        for predecessor in circuit_dag.predecessors(successor):
            if not circuit_dag.nodes[predecessor]['exec']:
                all_executed = False

        # If all predecessor of a successor have been executed this successor could be added to front layer
        if all_executed:
            addable_successors.append(successor)

    return addable_successors


def add_gates_to_circuit(node: str, new_circuit: Circuit, circuit_dag: nx.DiGraph, mapping: Mapping) -> None:
    """
    Adds the gates of a group to the circuit by associating the right physical qubits to them.

    Args:
        node (str): a node of the DAG
        new_circuit (Circuit): the new circuit respecting hardware constraints
        circuit_dag (networkx.DiGraph): the DAG representing the quantum circuit
        mapping (Mapping): the current mapping

    Return:
        None: nothing
    """

    # Iterate over ops and link them to the rights physical qubits
    for op in circuit_dag.nodes[node]['ops']:
        if len(op.qbits) == 1:
            op.qbits = [mapping.get_by_logical_index(op.qbits[0])]
        elif len(op.qbits) == 2:
            op.qbits = [
                mapping.get_by_logical_index(op.qbits[0]),
                mapping.get_by_logical_index(op.qbits[1])
            ]

    # Add ops to the new circuit
    new_circuit.ops.extend(circuit_dag.nodes[node]['ops'])


def get_swap_candidates(front_layer: List[str], chip_coupling_graph: nx.Graph, mapping: Mapping,
                        circuit_dag: nx.DiGraph) -> List[Tuple[int, int]]:
    """
    Extracts all swap that could be done with qubits of the front layer.

    Args:
        front_layer (List<str>): the list of nodes belonging to front layer
        chip_coupling_graph (nx.Graph): the coupling graph of the chip
        mapping (Mapping): the current mapping
        circuit_dag (networkx.DiGraph): the DAG representing the circuit

    Return:
        Tuple<int, int>: list of all pairs of qubits that can be swaped
    """

    # Init the list of candidates
    swap_candidates = []

    # Init the list of all qubits contained in the front layer
    logical_qbits_in_layer = []

    # Extract qubits of the front layer
    for node in front_layer:
        logical_qbits_in_layer.extend(circuit_dag.nodes[node]['ops'][0].qbits)
    # Delete duplicates in the list
    logical_qbits_in_layer = list(set(logical_qbits_in_layer))

    # Iterate over qubits of the front layer
    for logical_qbit in logical_qbits_in_layer:
        physical_qbit = mapping.get_by_logical_index(logical_qbit)
        physical_neighbors = chip_coupling_graph.neighbors(physical_qbit)

        for physical_neighbor in physical_neighbors:
            if logical_qbit != mapping.get_by_physical_index(physical_neighbor):
                swap_candidates.append((logical_qbit, mapping.get_by_physical_index(physical_neighbor)))

    return swap_candidates


def metric(front_layer: List[str], temp_mapping: Mapping, distances: dict, circuit_dag: nx.DiGraph) \
        -> int:
    """
    Computes the cost associated to a given swap.

    Args:
        front_layer (List<int>):
        temp_mapping (Mapping):
        distances (dict): two key dictionary containing distance between nodes
        circuit_dag (networkx.DiGraph): the DAG representing the quantum circuit

    Return:
        int: cost of the swap
    """

    score = 0

    for node in front_layer:
        physical_qbit_1 = temp_mapping.get_by_logical_index(circuit_dag.nodes[node]['ops'][0].qbits[0])
        physical_qbit_2 = temp_mapping.get_by_logical_index(circuit_dag.nodes[node]['ops'][0].qbits[1])
        score += distances[physical_qbit_1][physical_qbit_2]

    return score


def get_best_swap(score: dict) -> Tuple[int, int]:
    """
    Finds the best swap among all the candidates.

    Args:
        score (dict<Tuple<int, int>, int>): dictionary which keys are swaps and associated values are the
        corresponding score

    Return:
        Tuple<int, int>: a random swap among those which have the minimal score
    """

    # Get the minimal score
    min_score = min(score.values())

    # Init the list of best swap candidates
    best_swap_candidates = []

    # Iterate over swaps and check if it has the minimal score
    for swap in score.keys():
        if score[swap] == min_score:
            best_swap_candidates.append(swap)

    # Return a random candidate to limit tricky cases where the algorithm lockes itself into infinite loops
    return rd.choice(best_swap_candidates)


def link_final_measurement(job: Job, mapping: Mapping, nbqbits_topology: int, nbqbits_circuit: int) -> None:
    """
    During the process qubits have been shuffled. In the case of a measurement the measured qubits don't correspond
    anymore. This function modifies the physical qubits measured to ensure the logical qubits measured are the right
    ones.

    Args:
        job (Job): current job
        mapping (Mapping): the final mapping
        nbqbits_topology (int): the number of qubits of the chip
        nbqbits_circuit (int): the number of qubits used in the circuit

    Return:
        None: nothing
    """
    job.circuit.nbqbits = nbqbits_topology

    if job.type == ProcessingType.SAMPLE:
        right_qbits = []
        # Iterate over measured qubits whether the measure is partial or complete
        for qbit in job.qubits or range(nbqbits_circuit):
            # Find the physical qubit to measure
            right_qbits.append(mapping.get_by_logical_index(qbit))
        # Update the measured qubits list of the job
        job.qubits = right_qbits
    elif job.type == ProcessingType.OBSERVABLE:
        # If the number of qubits available on the topology is higher than qubits required by the circuit then the
        # number of qubits associated to the observable have to be updated to number of qubits in the topology
        job.observable.nbqbits = nbqbits_topology

        # Iterate over terms composing the observable
        new_terms_list = []

        for term in job.observable.terms:
            right_qbits = []
            for qbit in term.qbits:
                # Find the physical qubit to measure
                right_qbits.append(mapping.get_by_logical_index(qbit))

            # Update the measured qubits list of the term
            new_term = term.to_term()
            new_term.qbits = right_qbits
            new_terms_list.append(new_term)

        job.observable.set_terms(new_terms_list)


class Sabre(AbstractPlugin):
    """
    This plugin provides an implementation of Sabre algorithm to deal with difficulties encoutered when computing an
    algorithm on an hardware with particular contraints.
    """

    def compile(self, batch: Batch, hardware_specs: HardwareSpecs) -> Batch:
        """
        Iterates over jobs of the input batch and insert swaps in the circuit to make it executable according given
        hardware specs.

        Args:
            batch (Batch): the batch for execution
            hardware_specs (HardwareSpecs): the hardware specs

        Return:
            Batch: a new batch containing the modified quantum circuits
        """

        # Init the new batch which will be returned
        new_batch = copy.deepcopy(batch)

        if new_batch.meta_data is None:
            new_batch.meta_data = {}

        # A counter of the number of passes in the loop
        job_id = 0

        # If hardware specs have no constraints the algorithm is useless and is skipped
        if hardware_specs.topology.type == TopologyType.ALL_TO_ALL:
            return new_batch

        # For each job of the batch we adapt circuit to hardware constraints
        for job, chip_coupling_graph, nbqbits in iterate_job_and_topology(new_batch, hardware_specs):
            # Update job_id
            job_id += 1

            # Get job circuit
            circuit = job.circuit

            # Build distance matrix
            distances = dict(nx.all_pairs_shortest_path_length(chip_coupling_graph))

            # Transform the circuit into a Directed Acyclic Graph (DAG)
            circuit_dag = circuit_to_dag(circuit, nbqbits)

            # Init front layer
            front_layer = get_addable_successors('BEG', circuit_dag)

            # Define the default mapping
            mapping = Mapping(nbqbits)

            # Init output circuit copying original circuit and clearing operation list
            new_circuit = copy.deepcopy(circuit)
            new_circuit.ops = circuit_dag.nodes['BEG']['ops']

            # Add swaps while the front layer is not empty
            while len(front_layer) > 0:
                executable_nodes = []

                # Find executable gates in the front layer
                for node in front_layer:
                    if is_executable(node, chip_coupling_graph, mapping, circuit_dag):
                        executable_nodes.append(node)

                # Remove executable gates from the front layer and update the new circuit
                if executable_nodes:
                    for node in executable_nodes:
                        front_layer.remove(node)

                        # Update front layer
                        circuit_dag.nodes[node]['exec'] = True
                        front_layer.extend(get_addable_successors(node, circuit_dag))

                        # Complete new circuit
                        add_gates_to_circuit(node, new_circuit, circuit_dag, mapping)
                else:
                    score = {}
                    swap_candidates = get_swap_candidates(front_layer, chip_coupling_graph, mapping, circuit_dag)

                    # Find the best SWAP to insert
                    for swap in swap_candidates:
                        temp_mapping = copy.deepcopy(mapping)
                        temp_mapping.update(swap)

                        # Use of SABRE heuristic
                        score[swap] = metric(front_layer, temp_mapping, distances, circuit_dag)

                    # Find the best swap i.e the one with the minimal score
                    best_swap = get_best_swap(score)

                    # Update current mapping
                    mapping.update(best_swap)

                    # Insert the swap in the new circuit
                    swap_op = Op(
                        gate='SWAP',
                        qbits=[mapping.get_by_logical_index(best_swap[0]), mapping.get_by_logical_index(best_swap[1])],
                        type=OpType.GATETYPE
                    )
                    new_circuit.ops.append(swap_op)

            # Final processing: update the new job and add it to the new batch
            job.circuit = new_circuit
            link_final_measurement(job, mapping, nbqbits, circuit.nbqbits)
            new_batch.meta_data[f"JOB_MAPPING_{job_id}"] = str(mapping)

        return new_batch

    def post_process(self, batch_result):
        """
        Sabre plugin doesn't post process and so we return the batch_result.
        """

        return batch_result

    def do_post_processing(self):
        """
        Here we don't desire to post process the result and so we return False.
        """

        return False
