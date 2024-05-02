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

from speech_foundation_models.data.audio_iterators import LibrilightIterator


RESOURCES_DIR = f"{os.path.dirname(__file__)}/resources/"
VALID_CONFIG_YAML = f"""
basedir: {RESOURCES_DIR}librilight
"""


class LibrilightIteratorTestCase(unittest.TestCase):
    @patch(
        f"{LibrilightIterator.__module__}.{LibrilightIterator.__name__}._read_audio_file",
        return_value=np.array([1]))
    def test_basic_functionality(self, mock_read_audio_file):
        with patch('builtins.open', new_callable=mock_open, read_data=VALID_CONFIG_YAML):
            libri_iter = LibrilightIterator("config.yaml", 16000)
        iter_out = sorted(list(libri_iter), key=lambda x: x['id'])
        self.assertDictEqual({
            "id": "111_3_goofy_pluto_lb_64kb_0000",
            "raw": np.array([1]),
            "sampling_rate": 16000}, iter_out[0])
        args = sorted(call.args[0] for call in mock_read_audio_file.call_args_list)
        self.assertEqual(
            f"{RESOURCES_DIR}librilight/111/3/goofy_pluto_lb_64kb_0000.flac",
            args[0])

    @patch(
        f"{LibrilightIterator.__module__}.{LibrilightIterator.__name__}._read_audio_file",
        return_value=np.array([1]))
    def test_directory_with_last_slash(self, mock_read_audio_file):
        """
        Checks that everything works fine also in the case in which the basedir option in the
        config file ends with a backslash
        """
        with patch(
                'builtins.open',
                new_callable=mock_open,
                read_data=f"basedir: {RESOURCES_DIR}librilight/"):
            libri_iter = LibrilightIterator("config.yaml", 16000)
        iter_out = sorted(list(libri_iter), key=lambda x: x['id'])
        self.assertDictEqual({
            "id": "111_3_goofy_pluto_lb_64kb_0000",
            "raw": np.array([1]),
            "sampling_rate": 16000}, iter_out[0])
        args = sorted(call.args[0] for call in mock_read_audio_file.call_args_list)
        self.assertEqual(
            f"{RESOURCES_DIR}librilight/111/3/goofy_pluto_lb_64kb_0000.flac",
            args[0])

    def test_wrong_config(self):
        with self.assertRaises(AssertionError) as context:
            with patch('builtins.open', new_callable=mock_open, read_data="aaa: aaa\n"):
                _ = LibrilightIterator("config.yaml", 16000)
        self.assertTrue('should contain attribute `basedir`' in str(context.exception))

    @patch(
        f"{LibrilightIterator.__module__}.{LibrilightIterator.__name__}._read_audio_file",
        return_value=np.array([1]))
    def test_skipping_generated_samples(self, mock_read_audio_file):
        """
        Checks that the generated samples skipper works
        """
        with patch('builtins.open', new_callable=mock_open, read_data=VALID_CONFIG_YAML):
            libri_iter = LibrilightIterator("config.yaml", 16000)
        fake_gen_file_content = "id\tlanguage\ttext\n111_3_goofy_pluto_lb_64kb_0000\ten\taa\n"
        with patch('builtins.open', new_callable=mock_open, read_data=fake_gen_file_content):
            libri_iter.add_generated_samples_skipper("fake_out_file.tsv")
        iter_out = sorted(list(libri_iter), key=lambda x: x['id'])
        self.assertEqual(2, len(iter_out))
        self.assertEqual("111_3_goofy_pluto_lb_64kb_0001", iter_out[0]['id'])
        self.assertEqual("111_9_goofy_pluto_lb_64kb_0000", iter_out[1]['id'])

    @patch(
        f"{LibrilightIterator.__module__}.{LibrilightIterator.__name__}._read_audio_file")
    def test_samples_longer_than_30s(self, mock_read_audio_file):
        mock_read_audio_file.return_value = np.random.random_sample((16000 * 62))
        with patch('builtins.open', new_callable=mock_open, read_data=VALID_CONFIG_YAML):
            libri_iter = LibrilightIterator("config.yaml", 16000)
        fake_gen_file_content = "id\tlanguage\ttext\n111_3_goofy_pluto_lb_64kb_0000\ten\taa\n" \
                                "111_3_goofy_pluto_lb_64kb_0001\ten\taa\n" \
                                "111_9_goofy_pluto_lb_64kb_0000__0\ten\taa\n"
        with patch('builtins.open', new_callable=mock_open, read_data=fake_gen_file_content):
            libri_iter.add_generated_samples_skipper("fake_out_file.tsv")
        iter_out = sorted(list(libri_iter), key=lambda x: x['id'])
        self.assertEqual(2, len(iter_out))
        self.assertEqual("111_9_goofy_pluto_lb_64kb_0000__1", iter_out[0]['id'])
        self.assertEqual("111_9_goofy_pluto_lb_64kb_0000__2", iter_out[1]['id'])
        self.assertEqual(iter_out[0]['raw'].shape, (16000 * 30,))
        self.assertEqual(iter_out[1]['raw'].shape, (16000 * 2,))


if __name__ == '__main__':
    unittest.main()
