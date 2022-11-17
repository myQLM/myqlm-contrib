"""
@authors Simon Martiel <simon.martiel@atos.net>
@file qat/models/ibm.py
@namespace qat.models.ibm

This module contains the definition of IBM like devices and hardware models.
"""
from qat import devices
from .base import BaseModel


def h_with_u3(qbit, nqbits):
    """
    Implementation of a H gate with a U3 gate.

    This is used in the ObservableSplitter plugin.

    Arguments:
        qbit(int): the index of the qubit to perform the H gate on
        nqbits(int): the total number of qubits

    """
    import numpy as np
    from qat.lang.AQASM import QRoutine, AbstractGate

    def u3_matrix(theta, phi, lamb):
        """
        A function that generates the matrix implementation of the U3 gate.
        """
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lamb) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lamb)) * np.cos(theta / 2),
                ],
            ]
        )

    U3 = AbstractGate("U3", [float] * 3, arity=1, matrix_generator=u3_matrix)
    rout = QRoutine()
    wires = rout.new_wires(nqbits)
    U3(np.pi / 2, 0, np.pi)(wires[qbit])
    return rout


def sqrt_x_with_u3(qbit, nqbits):
    """
    Implementation of a SQRT(X) gate with a U3 gate.

    This is used in the ObservableSplitter plugin.

    Arguments:
        qbit(int): the index of the qubit to perform the H gate on
        nqbits(int): the total number of qubits

    """
    import numpy as np
    from qat.lang.AQASM import QRoutine, AbstractGate

    def u3_matrix(theta, phi, lamb):
        """
        A function that generates the matrix implementation of the U3 gate.
        """
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lamb) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lamb)) * np.cos(theta / 2),
                ],
            ]
        )

    U3 = AbstractGate("U3", [float] * 3, arity=1, matrix_generator=u3_matrix)
    rout = QRoutine()
    wires = rout.new_wires(nqbits)
    U3(np.pi / 2, -np.pi / 2, np.pi / 2)(wires[qbit])
    return rout


class IBMModel(BaseModel):
    """
    A factory constructing compilers, harware models, and noisy simulators for IBM-like devices.
    """

    HARDWARE_KEY = "ibm"

    MODELS = {
        "typical": {"error_1": 1e-3, "error_2": 1e-2, "measurement_error": 5e-2},
        "good": {"error_1": 1e-4, "error_2": 1e-3, "measurement_error": 5e-3},
        "excellent": {"error_1": 1e-5, "error_2": 1e-4, "measurement_error": 5e-4},
    }

    TOPOLOGIES = {
        "melbourne": devices.IBM_MELBOURNE,
        "melbourne_v2": devices.IBM_MELBOURNE_V2,
        "rueschlikon": devices.IBM_RUESCHLIKON,
        "singapore": devices.IBM_SINGAPORE,
        "tokyo": devices.IBM_TOKYO,
        "yorktown": devices.IBM_YORKTOWN,
    }

    BASIS_CHANGE = {"X": h_with_u3, "Y": sqrt_x_with_u3}
