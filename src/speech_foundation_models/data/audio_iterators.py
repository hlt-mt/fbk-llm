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
import math
import os
import pathlib
from typing import Any, Optional

import soundfile
import yaml


LOGGER = logging.getLogger(__name__)


class SamplesSkipper:
    """
    Interface for methods to determine whether a sample should be skipped or not.
    """
    def should_skip(self, sample_id):
        raise NotImplementedError(
            f"The instance {self.__class__} of SamplesSkipper must implement should_skip")


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
        self.samples_skipper: Optional[SamplesSkipper] = None
        self._additional_read_args = {}

    def add_generated_samples_skipper(self, output_file: str):
        self.samples_skipper = GeneratedSamplesSkipper(output_file)

    def should_skip_sample(self, sample_id):
        return self.samples_skipper is not None and self.samples_skipper.should_skip(sample_id)

    def _get_conf(self, attr: str) -> Any:
        assert attr in self.config, f"config file should contain attribute `{attr}`"
        return self.config[attr]

    def _read_audio_file(self, filename):
        audio_content, sample_rate = soundfile.read(
            filename, dtype='float32', always_2d=True, **self._additional_read_args)
        assert sample_rate == self.sampling_rate, \
            (f"Audio with sampling rate ({sample_rate}) not expected. "
             f"Expected rate was: {self.sampling_rate}")
        return audio_content[:, 0]  # return always the first channel

    def __iter__(self):
        raise NotImplementedError(
            f"Subclass {self.__class__} of AudioIterator must implement __iter__")


class GeneratedSamplesSkipper(SamplesSkipper):
    """
    A class that determines whether a sample has to be skipped or not based on whether it is
    already present in an output file or not.
    """
    def __init__(self, output_file: str):
        self.generated_ids = self._load_generated_ids(output_file)

    @staticmethod
    def _load_generated_ids(output_file):
        generated_ids = set()
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for segm in reader:
                generated_ids.add(segm['id'])
        return generated_ids

    def should_skip(self, sample_id):
        return sample_id in self.generated_ids


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
        self.truncate_exceeding_30s = self.config.get("truncate_exceeding_30s", True)
        self.basedir = os.path.dirname(self.tsv_segments)
        # As Whisper breaks if segments longer than 30s are fed, and in Voxpopuli some samples
        # contain a few frames more than 30s, ensure that we read at most 30s
        if self.truncate_exceeding_30s:
            self._additional_read_args["frames"] = self.sampling_rate * 30

    def __iter__(self):
        with open(self.tsv_segments, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                row_lang = row['event_id'].rsplit("_", maxsplit=1)[1].strip()
                row_year = row['event_id'][:4]
                if row_lang != self.lang:
                    continue
                row_id = f"{row['event_id']}_{row['segment_no']}"
                if self.should_skip_sample(row_id):
                    continue
                raw_content = self._read_audio_file(
                    f"{self.basedir}/{row_lang}/{row_year}/{row_id}.ogg")
                yield {
                    "id": row_id,
                    "raw": raw_content,
                    "sampling_rate": self.sampling_rate
                }


class LibrilightIterator(AudioIterator):
    """
    Iterator that returns the audios in the format used by Librilight.
    It needs as configuration (to be included in the YAML config file):
     - directory that contains the output generated by the `cut_by_vad.py` script of the Librilight
       repository (basedir)
    """
    def __init__(self, config: str, sampling_rate: int):
        super().__init__(config, sampling_rate)
        self.basedir = pathlib.Path(self._get_conf("basedir"))
        # Whisper supports only segments of less than 30s
        self.split_if_longer_than_30s = self.config.get("split_if_longer_than_30s", True)
        self.max_segment_length = self.sampling_rate * 30

    def __iter__(self):
        for speaker_path in self.basedir.iterdir():
            speaker_id = speaker_path.name
            for book_path in speaker_path.iterdir():
                book_id = book_path.name
                for flac_file in book_path.iterdir():
                    row_id = f"{speaker_id}_{book_id}_{flac_file.stem}"
                    if self.should_skip_sample(row_id):
                        continue
                    raw_content = self._read_audio_file(flac_file.as_posix())
                    content_length = raw_content.shape[0]
                    # in case of segments longer than 30 seconds, split them in 30s chunks
                    num_chunks = math.ceil(content_length / self.max_segment_length)
                    if self.split_if_longer_than_30s and num_chunks > 1:
                        for i in range(num_chunks):
                            new_id = row_id + f"__{i}"
                            if self.should_skip_sample(new_id):
                                continue
                            start_chunk = self.max_segment_length * i
                            end_chunk = min(self.max_segment_length * (i + 1), content_length)
                            yield {
                                "id": new_id,
                                "raw": raw_content[start_chunk:end_chunk],
                                "sampling_rate": self.sampling_rate
                            }
                    else:
                        yield {
                            "id": row_id,
                            "raw": raw_content,
                            "sampling_rate": self.sampling_rate
                        }
