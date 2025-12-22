"""
Cost function registry for YAML configuration loading.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from .base import CostFunction
from .penetration import PenetrationCost, SelfPenetrationCost
from .reference import ReferenceTrackingCost
from .temporal import AccelerationCost, JerkCost, VelocitySmoothnessCost


class CostRegistry:
    """
    Registry for cost function types.

    Enables creating costs from YAML configuration by type name.
    """

    _registry: Dict[str, Type[CostFunction]] = {}

    @classmethod
    def register(cls, name: str, cost_cls: Type[CostFunction]):
        """Register a cost function type."""
        cls._registry[name] = cost_cls

    @classmethod
    def get(cls, name: str) -> Type[CostFunction]:
        """Get a cost function type by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown cost type: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def create(cls, type_name: str, **kwargs) -> CostFunction:
        """Create a cost function instance from type name and kwargs."""
        cost_cls = cls.get(type_name)
        return cost_cls(**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """List all registered cost types."""
        return list(cls._registry.keys())


def register_cost(name: str) -> Callable:
    """Decorator to register a cost function type."""

    def decorator(cls: Type[CostFunction]) -> Type[CostFunction]:
        CostRegistry.register(name, cls)
        return cls

    return decorator


def create_cost(config: Dict[str, Any]) -> CostFunction:
    """
    Create a cost function from a configuration dict.

    Expected config format:
    {
        "type": "ReferenceTrackingCost",
        "name": "reference_tracking",  # optional
        "weight": 100.0,
        "enabled": true,
        "config": {...}  # cost-specific config
    }
    """
    type_name = config["type"]
    cost_cls = CostRegistry.get(type_name)

    return cost_cls(
        name=config.get("name", type_name.lower()),
        weight=config.get("weight", 1.0),
        enabled=config.get("enabled", True),
        config=config.get("config", {}),
    )


# Register built-in costs
CostRegistry.register("ReferenceTrackingCost", ReferenceTrackingCost)
CostRegistry.register("PenetrationCost", PenetrationCost)
CostRegistry.register("SelfPenetrationCost", SelfPenetrationCost)
CostRegistry.register("VelocitySmoothnessCost", VelocitySmoothnessCost)
CostRegistry.register("AccelerationCost", AccelerationCost)
CostRegistry.register("JerkCost", JerkCost)
