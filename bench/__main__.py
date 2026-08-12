"""Enables `python -m bench`, which works before the package is installed."""

from bench.cli import main

raise SystemExit(main())
