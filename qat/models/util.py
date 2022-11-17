"""
@authors Simon Martiel <simon.martiel@atos.net>
@file qat/models/util.py
@brief Some utility functions to build hardware models
@namespace qat.models.util

This module contains some generic methods to build hardware models
"""
import numpy as np
import networkx as nx
from qat.devices.util import build_device
from qat.hardware.models import make_depolarizing_hardware_model

from .administration import load_noisy_simulator

Noisy = load_noisy_simulator()


def build_hardware(connectivity, error_1=1e-2, error_2=1e-3, measurement_error=5e-2, **kwargs):
    """
    Builds a hardware model and a device object.

    Arguments:
        connectivity(:class:`qat.core.HardwareSpecs`,:class:`networkx.Graph`):
         the qubits connectivity
        error_1(float): the single qubit gate error rate
        error_2(float): the two qubit gate error rate
        measurement_error(float): the measurement error rate
        kwargs(keyword arguments): all additional argument are passed to the simulator

    Returns:
        a QPU
    """
    model = make_depolarizing_hardware_model(error_1, error_2)
    model.gates_specification.meas = np.array([measurement_error, 0, 0, 1 - measurement_error]).reshape((2, 2))
    if isinstance(connectivity, nx.Graph):
        connectivity = build_device(connectivity, "", "")
    if connectivity is not None:
        return connectivity.as_quameleon() | Noisy(hardware_model=model, **kwargs)
    return Noisy(hardware_model=model, **kwargs)
