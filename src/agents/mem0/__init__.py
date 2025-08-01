import importlib.metadata

try:
    __version__ = importlib.metadata.version("mem0ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from src.agents.mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from src.agents.mem0.memory.main import AsyncMemory, Memory  # noqa
