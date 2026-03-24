"""SCALeJ: Lennard-Jones Parameter Fitting via Condensed-Phase Volume-Scaling."""

try:
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
