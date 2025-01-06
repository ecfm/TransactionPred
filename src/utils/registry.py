from typing import Any, Callable, Dict

class Registry:
    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        def decorator(obj: Any) -> Any:
            self._registry[name] = obj
            return obj
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(f"{name} not found in registry")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

def create_registry() -> Registry:
    return Registry()