
import enum
import re
from typing import Self

import npc_ephys
import npc_session
import pydantic


class RecordingType(enum.Enum):
    AP = "AP"
    LFP = "LFP"
    NIDAQ = "NIDAQ"

PROBE_REGEX = r'Neuropix-PXI-\d{3}\.[^-]+-(?:AP|LFP)'
NIDAQ_REGEX = r'NI-DAQmx-\d{3}\.PXI-\d+'

class DatData(pydantic.BaseModel):

    dtype: str
    size: int
    path: str
    # the following fields are populated automatically (or attempted) if not provided
    session_id: str | None = None # this is the only one we cannot always determine automatically
    """AIND session ID: <platform>_<subject_id>_<date>_<time>"""
    subject_id: int = None # type: ignore[assignment]
    date: str = None # type: ignore[assignment]
    settings_xml_path: str = None # type: ignore[assignment]
    start_time: str = None # type: ignore[assignment]
    device_name: str = None # type: ignore[assignment]
    device_type: RecordingType = None # type: ignore[assignment]

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @pydantic.model_validator(mode='after')
    def try_get_session_id(self) -> Self:
        if self.session_id is None:
            try:
                self.session_id = npc_session.AINDSessionRecord(self.path)
            except ValueError: # cannot determine session ID from path
                pass
        return self

    @pydantic.model_validator(mode='after')
    def ensure_settings_xml_path(self) -> Self:
        if self.settings_xml_path is not None:
            return self
        import aind_ephys_cleanup.utils as utils
        self.settings_xml_path = utils.get_settings_xml_path(self.path).as_posix()
        return self

    @pydantic.model_validator(mode='after')
    def ensure_subject_id(self) -> Self:
        if self.subject_id is not None:
            return self
        subject_id = npc_session.extract_subject(self.path)
        if subject_id is None:
            raise ValueError(f"Could not determine subject ID from path: {self.path}")
        self.subject_id = subject_id
        return self

    @pydantic.model_validator(mode='after')
    def ensure_date(self) -> Self:
        if self.date is not None:
            return self
        date = npc_session.extract_isoformat_date(self.path)
        assert date is not None, f"Could not determine date from path: {self.path}"
        self.date = date
        return self

    @pydantic.model_validator(mode='after')
    def ensure_start_time(self) -> Self:
        if self.start_time is not None:
            return self
        assert self.settings_xml_path is not None, "settings_xml_path must be set to determine start_time"
        self.start_time = npc_ephys.get_settings_xml_data(self.settings_xml_path).start_time.isoformat()
        return self

    @pydantic.model_validator(mode='after')
    def ensure_device_name(self) -> Self:
        if self.device_name is not None:
            return self
        probe_match = re.search(PROBE_REGEX, self.path)
        if probe_match:
            self.device_name = probe_match.group(0)
        else:
            nidaq_match = re.search(NIDAQ_REGEX, self.path)
            if nidaq_match:
                self.device_name = nidaq_match.group(0)
            else:
                raise ValueError(f"Could not determine device name from path: {self.path}")
        return self

    @pydantic.model_validator(mode='after')
    def ensure_device_type(self) -> Self:
        if self.device_type is not None:
            return self
        if self.device_name.startswith("NI-DAQmx"):
            self.device_type = RecordingType.NIDAQ
        elif self.device_name.endswith("-AP"):
            self.device_type = RecordingType.AP
        elif self.device_name.endswith("-LFP"):
            self.device_type = RecordingType.LFP
        else:
            raise ValueError(f"Unknown device type for path: {self.path}")
        return self
    
class ZarrData(DatData):

    zarr_key: str
