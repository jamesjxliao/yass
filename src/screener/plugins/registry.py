from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from screener.plugins.base import Filter, Signal

logger = logging.getLogger(__name__)


class DuplicatePluginError(Exception):
    pass


def _has_protocol_shape(obj: object, protocol: type) -> bool:
    """Check if an object has all the required attributes of a protocol.

    We can't use issubclass() with runtime_checkable protocols that have
    non-method members in Python 3.12+, so we check structurally.
    """
    if protocol is Filter:
        return (
            hasattr(obj, "name")
            and hasattr(obj, "description")
            and callable(getattr(obj, "apply", None))
        )
    elif protocol is Signal:
        return (
            hasattr(obj, "name")
            and hasattr(obj, "description")
            and hasattr(obj, "higher_is_better")
            and callable(getattr(obj, "compute", None))
        )
    return False


def discover_plugins(directory: Path, protocol: type) -> dict[str, Any]:
    """Scan .py files in directory and return instances implementing the given protocol.

    Skips __init__.py. For each module, finds classes that match the protocol
    shape, instantiates them, and returns {plugin.name: plugin_instance}.
    """
    plugins: dict[str, Any] = {}
    directory = Path(directory)

    if not directory.exists():
        logger.warning("Plugin directory does not exist: %s", directory)
        return plugins

    for py_file in sorted(directory.glob("*.py")):
        if py_file.name == "__init__.py":
            continue

        module_name = f"plugin_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            logger.warning("Failed to load plugin module: %s", py_file, exc_info=True)
            continue

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)

            instance = None
            if isinstance(attr, type) and attr.__module__ == module_name:
                # It's a class defined in this module — try to instantiate
                try:
                    instance = attr()
                except Exception:
                    logger.warning(
                        "Failed to instantiate class %s from %s",
                        attr_name,
                        py_file,
                        exc_info=True,
                    )
                    continue
                if not _has_protocol_shape(instance, protocol):
                    instance = None
            elif not isinstance(attr, type) and _has_protocol_shape(attr, protocol):
                # It's already an instance with the right shape
                instance = attr

            if instance is not None:
                if instance.name in plugins:
                    raise DuplicatePluginError(
                        f"Duplicate plugin name '{instance.name}' "
                        f"found in {py_file} (already registered)"
                    )
                plugins[instance.name] = instance
                logger.debug("Discovered plugin: %s from %s", instance.name, py_file)

    return plugins


def discover_filters(directory: Path) -> dict[str, Filter]:
    return discover_plugins(directory, Filter)


def discover_signals(directory: Path) -> dict[str, Signal]:
    return discover_plugins(directory, Signal)
