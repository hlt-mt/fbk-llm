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

from speech_foundation_models.data.audio_iterators import YamlIterator


RESOURCES_DIR = f"{os.path.dirname(__file__)}/resources/"
VALID_CONFIG_YAML = f"""
basedir: wav_basedir
yaml_segment_definition: {RESOURCES_DIR}shas_sentence_def.yaml
"""


class YamlIteratorTestCase(unittest.TestCase):
    @patch(f"{YamlIterator.__module__}.{YamlIterator.__name__}._read_audio_file")
    def test_basic_functionality(self, mock_read_audio_file):
        mock_read_audio_file.return_value = np.random.random_sample((16000 * 70))
        with patch('builtins.open', new_callable=mock_open, read_data=VALID_CONFIG_YAML):
            yaml_iter = YamlIterator("config.yaml", 16000)
        iter_out = sorted(list(yaml_iter), key=lambda x: x['id'])
        self.assertEqual(5, len(iter_out))
        self.assertEqual('aa_0', iter_out[0]['id'])
        self.assertEqual((177457, ), iter_out[0]['raw'].shape)
        self.assertEqual(16000, iter_out[0]['sampling_rate'])
        self.assertEqual('aa_1', iter_out[1]['id'])
        self.assertEqual('bb_0', iter_out[2]['id'])
        self.assertEqual('bb_1', iter_out[3]['id'])
        self.assertEqual('cc_0', iter_out[4]['id'])
        args = sorted(call.args[0] for call in mock_read_audio_file.call_args_list)
        self.assertEqual("wav_basedir/aa.wav", args[0])

    def test_wrong_config(self):
        with self.assertRaises(AssertionError) as context:
            with patch('builtins.open', new_callable=mock_open, read_data="aaa: aaa\n"):
                _ = YamlIterator("config.yaml", 16000)
        self.assertTrue('should contain attribute `basedir`' in str(context.exception))

    @patch(f"{YamlIterator.__module__}.{YamlIterator.__name__}._read_audio_file")
    def test_skipping_generated_samples(self, mock_read_audio_file):
        """
        Checks that the generated samples skipper works
        """
        mock_read_audio_file.return_value = np.random.random_sample((16000 * 70))
        with patch('builtins.open', new_callable=mock_open, read_data=VALID_CONFIG_YAML):
            libri_iter = YamlIterator("config.yaml", 16000)
        fake_gen_file_content = "id\tlanguage\ttext\naa_1\ten\taa\n"
        with patch('builtins.open', new_callable=mock_open, read_data=fake_gen_file_content):
            libri_iter.add_generated_samples_skipper("fake_out_file.tsv")
        iter_out = sorted(list(libri_iter), key=lambda x: x['id'])
        self.assertEqual(4, len(iter_out))
        self.assertEqual("aa_0", iter_out[0]['id'])
        self.assertEqual("bb_0", iter_out[1]['id'])
        self.assertEqual("bb_1", iter_out[2]['id'])
        self.assertEqual("cc_0", iter_out[3]['id'])


if __name__ == '__main__':
    unittest.main()
