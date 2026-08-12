"""Custom Harbor agent adapters, one module per harness.

Each module exposes a ``BaseInstalledAgent`` subclass that Harbor loads via
``--agent harnesses.<module>:<Class>``. Harnesses that Harbor already ships an
agent for (e.g. hermes) need no module here -- only a block in registry.yaml.
"""
