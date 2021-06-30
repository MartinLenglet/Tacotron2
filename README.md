# WORK IN PROGRESS
README will be updated soon

# Tacotron2 (with WaveGlow) | GIPSA-LAB Version

Modified version of [Tacotron2 shared by NVIDIA].

This implementation includes all modifications described in [Link to SSW]. Visit our [website] for audio samples.

This repository gathers both the Tacotron2 end-to-end model and the neural vocoder [WaveGlow].

## Differences with [Tacotron2 shared by NVIDIA]
1. CSV format used for training, validation and test is the following:
    - No headers
    - One line by utterance
    - Line Format: Chapter_Name | start_utterance | end_utterance | text_utterance
    - Chapter_Name needs to correspond to the name of a .wav file (whose path is specified in hparams.py with "prefix_data")
    - start_utterrance and end_utterance are given in milliseconds in the corresponding .wav file
    - Both training and inference accept orthographic or/and phonetic inputs for text_utterance
    - An example is given with csv_files/ES_LMP_NEB_01_0001.csv
2. The gate loss correction is introduced (described in our paper [Link to SSW]):
    - a multiplying factor can be added to the gate loss error before back-propagation (specified in hparams.py with "loss_gate_weight")
    - Additional duration of ambient silence can be added at the end of each utterance, during which the end-of-sequence probability is set to 1 (specified in hparams.py with "end_silence_ms")
3. The possibility of exponentially decreasing the learning rate during training is added. 3 parameters are to be set (in hparams.py):
    - learning_rate_max: start value of the learning rate
    - learning_rate_min: end value of the learning rate
    - epoch_start: epoch number for which the learning rate starts to decrease
    - dr: decreasing rate
4. Particular treatment is added for end of paragraph marks "§":
    - when this paragraph mark is used within an utterance, this utterance is split at the character
    - each split is synthesized independently
    - all syntheses are concatenated
5. intermediate results can be save for each utterance (specified in hparams.py)
    - save_embeddings: save the embeddings matrix outputted by the Tacotron2 Encoder
    - save_alignments: save the alignment weights for each decoder timestep

## Pre-requisites
1. Python 3.6 (recommended)
2. Updated pip3

## Setup
1. Clone this repo: `git clone https://github.com/MartinLenglet/Tacotron2.git`
2. Install python requirements: `pip3 install -r requirements.txt`

## Download dataset
We have shared or dataset online: [Ressources for French TSS Blizzard Challenge](https://zenodo.org/record/4580406#%23.YI_qIyaxXmE).
These data include csv files using the format previously described. Audio files (.wav format, 22050Hz) are concatenated in chapters, refered in the first column of the csv files.
Half of the utterances have a phonetic transcript.
These data can be used as is with this implementation.

## Preprocessing data
Before training, audio wav files are to be converted into mel-spectrograms with respect to WaveGlow format. 
The script do_process_files.py converts all .wav files from the specified folder to another folder.
    - `python3 do_process_files.py`

## Download pre-trained models
This [Google Drive] repository includes:
1. Two trained Tacotron2 models:
    - tacotron2_statepredict.pt: [Tacotron 2] model trained on LJSpeech (English)
    - tacotron2_WAVEGLOW_mixed_inputs.pt: model P_g described in [Link to SSW] and trained on [our shared dataset] (French)
2. One trained WaveGlow model:
    - waveglow_NEB.pt: trained on [our shared dataset]

## Training
1. `python3 do_train.py --output_directory=out_path`
2. (using a pre-trained model) `python3 do_train.py --output_directory=out_path -c pre_trained_model_path --warm_start`

## Inference
For inference, CSV only needs 2 columns: Name_WAV | Text_Utterance
For teacher-forcing and ground-truth, use the format previously described.
1. (inference): `python3 do_syn.py -m model_path -e 'python3 inference.py -f tacotron2.txt -w waveglow_NEB.pt -o . -s 0.6' -o output_path --hparams nm_csv_test=test_csv_path`
2. (teacher-forcing): `python3 do_syn.py -m model_path -e 'python3 inference.py -f tacotron2.txt -w waveglow_NEB.pt -o . -s 0.6' -o output_path -p --hparams nm_csv_test=test_csv_path`
3. (ground-truth): `python3 do_syn.py -m model_path -e 'python3 inference.py -f tacotron2.txt -w waveglow_NEB.pt -o . -s 0.6' -o output_path -g --hparams nm_csv_test=test_csv_path`

## Acknowledgements

[Tacotron2 shared by NVIDIA]: (https://github.com/NVIDIA/tacotron2)
[Link to SSW]: ()
[website]: (http://www.gipsa-lab.fr/~martin.lenglet/segmentation_impact/index.html)
[WaveGlow]: (https://ieeexplore.ieee.org/document/8683143)
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[our shared dataset]: https://zenodo.org/record/4580406#.YI_qIyaxXmE
[Google Drive]: https://drive.google.com/drive/folders/1au5v_69FdK62GcwixBVU7ftBNErnrovt?usp=sharing