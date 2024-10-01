# MOSEL: 950,000 Hours of Speech Data for Open-Source Speech Foundation Model Training on EU Languages

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