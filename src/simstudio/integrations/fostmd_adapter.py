"""Adapter for the FOSTMD external processing pipeline.

Status: **STUB** — export methods return empty dicts until the ``fostmd``
package is installed and the integration is implemented.

Usage:
    When the ``fostmd`` Python package is available in the environment,
    implement :meth:`FostmdAdapter.export_video_artifacts` and
    :meth:`FostmdAdapter.export_das_artifacts` to pass the simulator
    recording buffer to the pipeline.

TODO:
    - Implement export_video_artifacts(): convert sim_buffer frames to video
      frames compatible with the FOSTMD ingestion format.
    - Implement export_das_artifacts(): serialise DAS trace data from
      sim_buffer into FOSTMD-compatible HDF5 / CSV artefacts.
    - Add unit tests once the FOSTMD API is stable.
"""

from typing import Any, Dict, Optional


def is_fostmd_available() -> bool:
    """Return ``True`` if the ``fostmd`` package can be imported."""
    try:
        import fostmd  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


class FostmdAdapter:
    """Thin wrapper around the FOSTMD external pipeline.

    Args:
        main_dir: Path to the FOSTMD main directory, or ``None`` to use
                  the default configured in the ``fostmd`` package.
    """

    def __init__(self, main_dir: Optional[str] = None) -> None:
        self.main_dir = main_dir

    def export_video_artifacts(self, sim_buffer: Any) -> Dict[str, str]:
        """Convert *sim_buffer* to FOSTMD video artefacts.

        Args:
            sim_buffer: The recorded simulation event buffer.

        Returns:
            A dict mapping artefact names to file paths.
            Currently returns ``{}`` (not yet implemented).
        """
        # TODO: implement when fostmd package is available.
        return {}

    def export_das_artifacts(self, sim_buffer: Any) -> Dict[str, str]:
        """Convert *sim_buffer* DAS traces to FOSTMD artefacts.

        Args:
            sim_buffer: The recorded simulation event buffer.

        Returns:
            A dict mapping artefact names to file paths.
            Currently returns ``{}`` (not yet implemented).
        """
        # TODO: implement when fostmd package is available.
        return {}
