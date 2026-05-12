# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys
import types
from unittest.mock import MagicMock

class MockModule(MagicMock):
    def __getattr__(self, name):
        return MagicMock()

# 1. Mock problematic modules before they are imported
for mod in ['torch', 'numpy', 'vllm', 'msgpack', 'msgspec', 'aiter', 'transformers', 'zmq', 'xxhash', 'fastapi', 'psutil', 'protobuf', 'uvicorn', 'aiohttp', 'datasets', 'openpyxl', 'tqdm', 'quart']:
    sys.modules[mod] = MockModule()

# 2. Stub the base module before importing factory
_atom_base = types.ModuleType("atom.kv_transfer.disaggregation.base")
class KVConnectorBase: pass
class KVConnectorSchedulerBase: pass
_atom_base.KVConnectorBase = KVConnectorBase
_atom_base.KVConnectorSchedulerBase = KVConnectorSchedulerBase
sys.modules["atom.kv_transfer.disaggregation.base"] = _atom_base

# 3. Prevent atom.__init__ from doing anything
_atom = types.ModuleType("atom")
_atom.__path__ = ["atom"]
sys.modules["atom"] = _atom

import pytest
from unittest.mock import patch
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

class MockWorker:
    def __init__(self, config):
        self.config = config

class MockScheduler:
    def __init__(self, config):
        self.config = config

@pytest.fixture
def clean_registry():
    """Fixture to ensure the registry is restored after each test."""
    original_registry = KVConnectorFactory._registry.copy()
    yield
    KVConnectorFactory._registry = original_registry

def test_register(clean_registry):
    KVConnectorFactory.register(
        "test_backend",
        worker_module="test_mod",
        worker_class="Worker",
        scheduler_module="test_mod",
        scheduler_class="Scheduler"
    )
    assert "test_backend" in KVConnectorFactory._registry
    assert KVConnectorFactory._registry["test_backend"] == {
        "worker_module": "test_mod",
        "worker_class": "Worker",
        "scheduler_module": "test_mod",
        "scheduler_class": "Scheduler"
    }

def test_create_connector_unknown_backend():
    config = MagicMock()
    config.kv_transfer_config = {"kv_connector": "non_existent"}
    with pytest.raises(ValueError, match="Unknown KV connector backend"):
        KVConnectorFactory.create_connector(config)

def test_create_connector_unknown_role():
    config = MagicMock()
    # "moriio" is registered by default
    config.kv_transfer_config = {"kv_connector": "moriio"}
    with pytest.raises(ValueError, match="Unknown role"):
        KVConnectorFactory.create_connector(config, role="invalid_role")

@patch("importlib.import_module")
def test_create_connector_worker(mock_import, clean_registry):
    mock_mod = MagicMock()
    mock_mod.TestWorker = MockWorker
    mock_import.return_value = mock_mod

    KVConnectorFactory.register(
        "test_worker",
        worker_module="test_mod",
        worker_class="TestWorker",
        scheduler_module="test_mod",
        scheduler_class="TestScheduler"
    )

    config = MagicMock()
    config.kv_transfer_config = {"kv_connector": "test_worker"}

    connector = KVConnectorFactory.create_connector(config, role="worker")

    assert isinstance(connector, MockWorker)
    assert connector.config == config
    mock_import.assert_called_with("test_mod")

@patch("importlib.import_module")
def test_create_connector_scheduler(mock_import, clean_registry):
    mock_mod = MagicMock()
    mock_mod.TestScheduler = MockScheduler
    mock_import.return_value = mock_mod

    KVConnectorFactory.register(
        "test_scheduler",
        worker_module="test_mod",
        worker_class="TestWorker",
        scheduler_module="test_mod",
        scheduler_class="TestScheduler"
    )

    config = MagicMock()
    config.kv_transfer_config = {"kv_connector": "test_scheduler"}

    connector = KVConnectorFactory.create_connector(config, role="scheduler")

    assert isinstance(connector, MockScheduler)
    assert connector.config == config
    mock_import.assert_called_with("test_mod")
