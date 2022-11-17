"""
@authors Simon Martiel <simon.martiel@atos.net>
@file qat/models/administration.py
@brief Administrative functions to inspect environment
@namespace qat.models.administration
"""
import importlib


def load_plugin(plugin_name: str):
    """
    By default, attempts a plugin from qlmaas.
    If it fails, attempts to load it from the local environment.

    Arguments:
        plugin_name (str): a plugin name
    """
    try:
        plugins = importlib.import_module("qlmaas.plugins")
        return getattr(plugins, plugin_name)
    except ImportError:
        plugins = importlib.import_module("qat.plugins")
        return getattr(plugins, plugin_name)


def load_noisy_simulator():
    """
    Attempts a noisy simulator from qlmaas.
    If it fails, attempts to load it from the local environment.
    """
    try:
        plugins = importlib.import_module("qlmaas.qpus")
        return getattr(plugins, "NoisyQProc")
    except ImportError:
        plugins = importlib.import_module("qat.qpus")
        return getattr(plugins, "NoisyQProc")
