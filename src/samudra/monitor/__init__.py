"""Continuous monitoring — the loop that turns the pipeline into a service.

Everything else in SAMUDRA is a stage you invoke. This is the part that runs on
its own, watches for new data, and processes it unattended.
"""

from samudra.monitor.service import (  # noqa: F401
    MONITOR_STATUS_PATH,
    scan_once,
    read_status,
    run,
)
