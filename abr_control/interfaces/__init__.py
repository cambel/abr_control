from .vrep import VREP
try:
    from .ipygame import PyGame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False