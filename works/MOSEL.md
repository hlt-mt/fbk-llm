# MOSEL: 950,000 Hours of Speech Data for Open-Source Speech Foundation Model Training on EU Languages

## Data Download and Preparation

Download the VoxPopuli data following the [official README for downloading the unlabelled data](https://github.com/facebookresearch/voxpopuli/blob/main/README.md#unlabelled-data). Follow the README to obtain the data for all the language codes (`--subset`) listed in the [MOSEL HuggingFace release](https://huggingface.co/datasets/FBK-MT/mosel#dataset-statistics-in-hours). Once segmented using the default script, we converted the `.ogg` files into `.wav` files with [ffmpeg](https://ffmpeg.org/) using `ffmpeg -i OGG_FILE -ac 1 -ar 16000 WAV_FILE`.

Download the LibriLight data following the [Data Preparation and Download README](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md) with the default parameters. Then, for the [Segmentation step](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md#1b-segmenting), use `python cut_by_vad.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --target_len_sec=30` to obtain the audio files compatible with what is present in MOSEL.

## Data Transcription

All the transcripts of the MOSEL dataset were obtained with the code in this repository, using the `whisper-gen` command.
Below we describe the steps to replicate our procedure and results.

The procedure was executed using `huggingface==4.38.2` on A100 64GB GPUs.
We used beam size 5. To speed up the inference process, we set the batch size to 16 and
enabled FlashAttention that can be installed with the following command:

```
pip install flash-attn --no-build-isolation
```

Depending on the target language, the transcription process generated ~40-50K samples per day.

Overall, an example command (for the Estonian split of VoxPopuli) is this:

```
whisper-gen --logging-freq 100 --audio-iterator voxpopuli \
   --audio-iterator-config config_asr_voxpopuli_et.yaml \
   -o voxpopuli.et.tsv --beam-size 5 --use-flash-attention --batch-size 16 --source-lang et
```

where `config_asr_voxpopuli_et.yaml` contains:

```
lang: et
tsv_segments: voxpopuli/et/unlabelled_data/unlabelled_v2_et.tsv
```

## Citation

```
@inproceedings{mosel,
  title = {{MOSEL: 950,000 Hours of Speech Data for Open-Source Speech Foundation Model Training on EU Languages}},
  author = {Marco Gaido and Sara Papi and Luisa Bentivogli and Alessio Brutti and Mauro Cettolo and Roberto Gretter and Marco Matassoni and Mohamed Nabihand Matteo Negri},
  booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2024",
  address = "Miami, United States",
  publisher = "Association for Computational Linguistics",
}
```