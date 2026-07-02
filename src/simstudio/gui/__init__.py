"""SimStudio GUI layer.

This sub-package contains the Tkinter desktop application.  It is intentionally
separate from the core simulation library (``simstudio``) so that the engine,
models, and sensor modules can be imported in headless / server environments
without pulling in any GUI dependencies.

Entry point::

    from simstudio.gui.app import run
    run()

Or via the project launcher::

    python scripts/run_app.py
"""
