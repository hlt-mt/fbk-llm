# Copyright 2024 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import csv
import logging
import os
from typing import Any

import soundfile
import yaml


LOGGER = logging.getLogger(__name__)


class AudioIterator:
    """
    Base class that returns an iterator of dictionaries representing an audio file to be processed
    with Whisper. It supports reading a config file in YAML format to enable customized
    configuration of the subclasses.
    """
    def __init__(self, config: str, sampling_rate: int):
        with open(config) as f:
            self.config = yaml.safe_load(f)
        LOGGER.info(f"Parsed AudioIterator config ({config}): {self.config}")
        self.sampling_rate = sampling_rate

    def __iter__(self):
        raise NotImplementedError(
            f"Subclass {self.__class__} of AudioIterator must implement __iter__")


class VoxpopuliIterator(AudioIterator):
    """
    Iterator that returns the audios in the format used by Voxpopuli.
    It needs as configuration (to be included in the YAML config file):
     - the language to consider (lang);
     - the path to the Voxpopuli TSV defining the segments (tsv_segments)
    """
    def __init__(self, config: str, sampling_rate: int):
        super().__init__(config, sampling_rate)
        self.lang = self._get_conf("lang")
        self.tsv_segments = self._get_conf("tsv_segments")
        self.basedir = os.path.dirname(self.tsv_segments)

    def _get_conf(self, attr: str) -> Any:
        assert attr in self.config, f"config file should contain attribute `{attr}`"
        return self.config[attr]

    def _read_audio_file(self, filename):
        audio_content, sample_rate = soundfile.read(
            filename, dtype='float32', always_2d=True)
        assert sample_rate == self.sampling_rate, \
            (f"Audio with sampling rate ({sample_rate}) not expected. "
             f"Expected rate was: {self.sampling_rate}")
        return audio_content[:, 0]  # return always the first channel

    def __iter__(self):
        with open(self.tsv_segments, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                row_lang = row['event_id'].split("_")[1].strip()
                row_year = row['event_id'][:4]
                if row_lang != self.lang:
                    continue
                row_id = f"{row['event_id']}_{row['segment_no']}"
                raw_content = self._read_audio_file(
                    f"{self.basedir}/{row_lang}/{row_year}/{row_id}.ogg")
                yield {
                    "id": row_id,
                    "raw": raw_content,
                    "sampling_rate": self.sampling_rate
                }
