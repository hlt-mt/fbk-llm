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
import csv
import logging
import os

import torch

from transformers import WhisperForConditionalGeneration, AutoProcessor, pipeline

from speech_foundation_models.inference.asr_args import add_gen_args, add_logging_args, \
    add_whisper_args


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOG_LEVEL", "INFO").upper())
LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    LOGGER.info(f"Parsed args: {args}")
    assert args.max_tokens <= 444, \
        f"Invalid max-tokens ({args.max_tokens}). " \
        "Whisper max_length is 448 and we reserve 4 tokens for the prefix."
    processor = AutoProcessor.from_pretrained(args.hf_model_name)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        args.hf_model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        use_flash_attention_2=args.use_flash_attention)
    model.eval()
    if torch.cuda.is_available() and not args.cpu:
        model = model.cuda()
    generate_kwargs = {
        "task": args.task,
        "no_timestamps_token_id": True,
        "num_beams": args.beam_size}
    if hasattr(args, "source_lang"):
        generate_kwargs["language"] = args.source_lang
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
        return_timestamps=False,
        device=model.device,
        torch_dtype=torch_dtype,
        return_language=True,
        generate_kwargs=generate_kwargs
    )
    audio_iterator = args.audio_iterator(
        args.audio_iterator_config, processor.feature_extractor.sampling_rate)

    with torch.no_grad():
        i = 0
        with open(args.output, 'w') as f_w:
            writer = csv.DictWriter(
                f_w, ["id", "language", "text"], delimiter='\t')
            writer.writeheader()
            LOGGER.info("Starting transcription")
            for output_row in transcriber(iter(audio_iterator)):
                i += 1
                if i % args.logging_freq == 0:
                    LOGGER.info(f"Processed {i} samples")
                assert len(output_row["chunks"]) == 1, \
                    f"The following output has more than one chunk: {output_row}"
                writer.writerow({
                    "id": output_row["id"][0],
                    "language": output_row["chunks"][0]["language"],
                    "text":  output_row["text"]})
    LOGGER.info(f"Transcription completed. Processed {i} samples.")


def cli_script():
    """
    Generates the output for the audios returned by the specified audio iterator and stores them
    into an output TSV which contains the ID of the sentence (returned by the audio iterator),
    the predicted language, and the transcription generated by Whisper.

    As the size of the iterators is unknown in advance we cannot report a percentage of completion,
    so we report only the advancement in terms of processed sentences.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_gen_args(parser)
    add_logging_args(parser)
    add_whisper_args(parser)
    parser.add_argument(
        '--output', '-o', type=str, required=True, help="the path to the output file")
    main(parser.parse_args())


if __name__ == "__main__":
    cli_script()
