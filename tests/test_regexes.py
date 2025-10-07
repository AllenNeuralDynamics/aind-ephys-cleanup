import re
import pytest

from aind_ephys_cleanup import models

def test_probe_regex():
    # Valid cases
    assert re.fullmatch(models.PROBE_REGEX, "Neuropix-PXI-100.ProbeD-AP")
    assert re.fullmatch(models.PROBE_REGEX, "Neuropix-PXI-123.XYZ-LFP")
    assert re.fullmatch(models.PROBE_REGEX, "Neuropix-PXI-999.ABC123-AP")
    # Invalid cases
    assert not re.fullmatch(models.PROBE_REGEX, "Neuropix-PXI-10.ProbeD-AP")  # only 2 digits for RecordNode
    assert not re.fullmatch(models.PROBE_REGEX, "Neuropix-PXI-100.ProbeD-XYZ")  # not AP or LFP
    assert not re.fullmatch(models.PROBE_REGEX, "Neuropix-PXI-100.ProbeD-APX")  # APX not allowed

def test_nidaq_regex():
    # Valid cases
    assert re.fullmatch(models.NIDAQ_REGEX, "NI-DAQmx-101.PXI-6133")
    assert re.fullmatch(models.NIDAQ_REGEX, "NI-DAQmx-999.PXI-1")
    assert re.fullmatch(models.NIDAQ_REGEX, "NI-DAQmx-000.PXI-123456")
    # Invalid cases
    assert not re.fullmatch(models.NIDAQ_REGEX, "NI-DAQmx-10.PXI-6133")  # only 2 digits for RecordNode
    assert not re.fullmatch(models.NIDAQ_REGEX, "NI-DAQmx-101.PXI-")  # missing digits
    assert not re.fullmatch(models.NIDAQ_REGEX, "NI-DAQmx-101.PXI-abc")  # non-digit
