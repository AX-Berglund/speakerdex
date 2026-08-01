"""speakerdex: persistent speaker identity across files.

A local voice registry that sits on top of any diarization pipeline and makes
sure the same voice gets the same name — across episodes, recordings and
projects.
"""

__version__ = "0.1.0"

from .matcher import MatchConfig
from .pipeline import ProcessConfig, enroll_from_audio, process_file
from .registry import Registry
from .types import Segment

__all__ = [
    "MatchConfig",
    "ProcessConfig",
    "Registry",
    "Segment",
    "enroll_from_audio",
    "process_file",
    "__version__",
]
