
import enum
import re

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

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @pydantic.computed_field()
    @property
    def session_id(self) -> npc_session.AINDSessionRecord | None:
        try:
            return npc_session.AINDSessionRecord(self.path)
        except ValueError:
            return None

    @pydantic.computed_field()
    @property
    def subject_id(self) -> int:
        subject_id = npc_session.extract_subject(self.path)
        if subject_id is None:
            raise ValueError(f"Could not determine subject ID from path: {self.path}")
        return subject_id

    @pydantic.computed_field()
    @property
    def date(self) -> str:
        date = npc_session.extract_isoformat_date(self.path)
        assert date is not None, f"Could not determine date from path: {self.path}"
        return date

    @pydantic.computed_field()
    @property
    def settings_xml_path(self) -> str:
        import aind_ephys_cleanup.utils as utils
        return utils.get_settings_xml_path(self.path).as_posix()

    @pydantic.computed_field()
    @property
    def start_time(self) -> str:
        return npc_ephys.get_settings_xml_data(self.settings_xml_path).start_time.isoformat()

    @pydantic.computed_field()
    @property
    def device_name(self) -> str:
        probe_match = re.search(PROBE_REGEX, self.path)
        if probe_match:
            return probe_match.group(0)
        nidaq_match = re.search(NIDAQ_REGEX, self.path)
        if nidaq_match:
            return nidaq_match.group(0)
        raise ValueError(f"Could not determine device name from path: {self.path}")

    @pydantic.computed_field()
    @property
    def device_type(self) -> RecordingType:
        if self.device_name.startswith("NI-DAQmx"):
            return RecordingType.NIDAQ
        if self.device_name.endswith("-AP"):
            return RecordingType.AP
        elif self.device_name.endswith("-LFP"):
            return RecordingType.LFP
        raise ValueError(f"Unknown device type for path: {self.path}")



class ZarrData(DatData):

    segment_name: str
