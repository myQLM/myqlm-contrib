"""
@authors Simon Martiel <simon.martiel@atos.net>
@file qat/models/base.py
@brief The base model for all hardware factories
@namespace qat.models.base
"""
from .administration import load_plugin
from .util import build_hardware

NISQCompiler = load_plugin("NISQCompiler")
ObservableSplitter = load_plugin("ObservableSplitter")


class BaseModel:
    """
    The base class for hardware models
    """

    HARDWARE_KEY = None
    TOPOLOGIES = {}
    MODELS = {}
    BASIS_CHANGE = {}

    @classmethod
    def get_sampler(cls):
        """
        Builds an observable sampler plugin (i.e. an ObservableSplitter) that uses the target gate set to perform the
        adequate basis changes.
        """
        # return cls.OBS_SPLITTER
        return ObservableSplitter(
            splitting_method="coloring",
            x_basis_change=cls.BASIS_CHANGE["X"],
            y_basis_change=cls.BASIS_CHANGE["Y"],
        )

    @classmethod
    def get_compiler(cls, *args, **kwargs):
        """
        Builds a compiler targetting some hardware.

        This is a wrapper that builds a NISQCompiler object.
        Have a look at the documentation of this plugin for more details.

        Arguments:
            args: all arguments are passed to NISQCompiler's constructor
            kwargs: all arguments are passed as backend optimizer options to the compiler.

        Returns:
            a Plugin ready to use
        """
        return NISQCompiler(*args, compiler_options=kwargs, target_gate_set=cls.HARDWARE_KEY)

    @classmethod
    def get_qpu(cls, topology=None, model="typical", sampler=True):
        """
        Builds a qpu emulating some hardware.

        Arguments:
            topology(str or HardwareSpecs): the topology of the processor. If a str, it should be a key
              of cls.TOPOLOGIES
            model(str or dict): the error rates of the processor. If a str, it should be a key
              of cls.MODELS. Otherwise it should be a dict containing keys "eps1", "eps2", "meas" corresponding
              to 1-qubit error rate, 2-qubit error rate and measurement error rate.
            sampler(optional, bool): if set to True, attaches an observable sampler plugin to the QPU. Defaults to True.

        Returns:
            a Plugin ready to use
        """
        if isinstance(topology, str) and topology not in cls.TOPOLOGIES:
            raise ValueError(f"Unknown topology {topology}")
        if isinstance(topology, str):
            topology = cls.TOPOLOGIES[topology]
        if isinstance(model, str):
            if model not in cls.MODELS:
                raise ValueError(f"Unknown model {model}")
            model = cls.MODELS[model]
        if sampler:
            return cls.get_sampler() | build_hardware(topology, **model)
        return build_hardware(topology, **model)
