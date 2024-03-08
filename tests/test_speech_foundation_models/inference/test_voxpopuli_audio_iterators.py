# Copyright 2024 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os.path
import unittest
from unittest.mock import patch, mock_open

import numpy as np

from speech_foundation_models.data.audio_iterators import VoxpopuliIterator

VALID_CONFIG_YAML = f"""
lang: it
tsv_segments: {os.path.dirname(__file__)}/resources/voxpopuli.tsv
"""


class VoxpopuliIteratorTestCase(unittest.TestCase):
    @patch(
        f"{VoxpopuliIterator.__module__}.{VoxpopuliIterator.__name__}._read_audio_file",
        return_value=[])
    def test_basic_functionality(self, mock_read_audio_file):
        with patch('builtins.open', new_callable=mock_open, read_data=VALID_CONFIG_YAML):
            vox_iter = VoxpopuliIterator("config.yaml", 16000)
        iterator = iter(vox_iter)
        self.assertDictEqual({
            "id": "20200113-0900-PLENARY_it_0",
            "raw": [],
            "sampling_rate": 16000}, next(iterator))
        args, _ = mock_read_audio_file.call_args
        self.assertEqual(
            f"{os.path.dirname(__file__)}/resources/it/2020/20200113-0900-PLENARY_it_0.ogg",
            args[0])

    @patch(
        f"{VoxpopuliIterator.__module__}.{VoxpopuliIterator.__name__}._read_audio_file",
        return_value=[])
    def test_language_filtering(self, mock_read_audio_file):
        with patch('builtins.open', new_callable=mock_open, read_data=VALID_CONFIG_YAML):
            vox_iter = VoxpopuliIterator("config.yaml", 16000)
        iterator = iter(vox_iter)
        generated_items = list(iterator)
        self.assertEqual(2, len(generated_items))

    def test_wrong_config(self):
        with self.assertRaises(AssertionError) as context:
            with patch('builtins.open', new_callable=mock_open, read_data="tsv_segments: aaa\n"):
                _ = VoxpopuliIterator("config.yaml", 16000)
        self.assertTrue('should contain attribute `lang`' in str(context.exception))

    @patch(
        f"{VoxpopuliIterator.__module__}.{VoxpopuliIterator.__name__}._read_audio_file",
        return_value=np.array([1]))
    def test_skipping_generated_samples(self, mock_read_audio_file):
        with patch('builtins.open', new_callable=mock_open, read_data=VALID_CONFIG_YAML):
            vox_iter = VoxpopuliIterator("config.yaml", 16000)
        fake_gen_file_content = "id\tlanguage\ttext\n20200113-0900-PLENARY_it_0\tit\taa\n"
        with patch('builtins.open', new_callable=mock_open, read_data=fake_gen_file_content):
            vox_iter.add_generated_samples_skipper("fake_out_file.tsv")
        iter_out = list(vox_iter)
        self.assertEqual(1, len(iter_out))
        self.assertEqual("20200113-0900-PLENARY_it_1", iter_out[0]['id'])


if __name__ == '__main__':
    unittest.main()
