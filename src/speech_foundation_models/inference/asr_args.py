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
import argparse
import inspect

from speech_foundation_models.data import audio_iterators


def collect_audio_iterators():
    iterators = {}
    for name, class_obj in inspect.getmembers(audio_iterators, inspect.isclass):
        if issubclass(class_obj, audio_iterators.AudioIterator) \
                and class_obj != audio_iterators.AudioIterator:
            actual_name = name.split(".")[-1].replace("Iterator", "").strip().lower()
            iterators[actual_name] = class_obj
    return iterators


class AudioIteratorArgparse(argparse.Action):
    """
    Custom argparse object that automatically looks for the `audio_iterators.AudioIterator` classes
    and returns the selected one.
    """
    AVAILABLE_ITERATORS = collect_audio_iterators()

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.AVAILABLE_ITERATORS[values])


def add_gen_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--hf-model-name', type=str, default="openai/whisper-large-v3",
        help="the Huggingface name of the model to use.")
    parser.add_argument(
        '--audio-iterator',
        action=AudioIteratorArgparse,
        choices=AudioIteratorArgparse.AVAILABLE_ITERATORS.keys(),
        required=True,
        help="audio iterator to be used to read audios")
    parser.add_argument(
        '--audio-iterator-config', '-y', type=str, required=True,
        help="a YAML file that contains the custom configurations for the audio iterator")
    parser.add_argument(
        '--batch-size', '-bs', type=int, default=1,
        help='batch size to use in the inference phase.')
    parser.add_argument(
        '--max-tokens', '-t', type=int, default=444,
        help='maximum number of generated tokens after which the generation stops.')
    parser.add_argument(
        '--beam-size', '-bm', type=int, default=1,
        help='beam size to use in the beam search.')
    parser.add_argument(
        '--cpu', '-c', default=False, action='store_true',
        help='if use CPU for generate even though GPUs are available.')


def add_whisper_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--task', type=str, default="transcribe", choices=["transcribe", "translate"],
        help="the task to perform. Can be either `transcribe` (default) or `translate`")
    parser.add_argument(
        '--source-lang', type=str,
        help="the language of the audio to transcribe/translate. "
             "If not set, it is predicted automatically")


def add_logging_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--logging-freq', '-lf', type=int, default=100,
        help='the frequency (in numer of samples) at which print logs.')
