# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import yaml
from yacs.config import CfgNode


def readsplits(splits, sample_rate=24000, n_shift=300):
    splits = [split.split(" ") for split in splits[2:]]
    phones = []
    ends = []
    accum = 0
    for phone, note, _, duration, slur in zip(*splits):
        accum += float(duration)
        if slur == "1":
            phone = "<slur>"
        phone += "<note:" + note + ">"
        phones.append(phone)
        ends.append(accum)
    frame_pos = librosa.time_to_frames(ends, sr=sample_rate, hop_length=n_shift)
    durations = np.diff(frame_pos, prepend=0)
    assert len(durations) == len(phones)
    results = ""
    for (p, d) in zip(phones, durations):
        results += p + " " + str(d) + " "
    return results.strip()


def gen_duration_for_svs(inputdir, output, sample_rate=24000, n_shift=300):
    # key: utt_id, value: (speaker, phn_durs)
    durations_dict = {}
    list_dir = os.listdir(inputdir)
    speakers = [dir for dir in list_dir if os.path.isdir(inputdir / dir)]
    for speaker in speakers:
        subdir = inputdir / speaker
        for file in os.listdir(subdir):
            if file.endswith(".txt"):
                with open(
                        os.path.join(subdir, file), "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                for line in lines:
                    splits = line.split('|')
                    name = splits[0]
                    durations_dict[name] = (speaker, readsplits(
                        splits, sample_rate=sample_rate, n_shift=n_shift))
    with open(output, "w") as wf:
        for name in sorted(durations_dict.keys()):
            wf.write(name + "|" + durations_dict[name][0] + "|" +
                     durations_dict[name][1] + "\n")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess transcription files for the datasets of SVS task."
    )
    parser.add_argument(
        "--inputdir",
        default=None,
        type=str,
        help="directory to transcription files.")
    parser.add_argument(
        "--output", type=str, required=True, help="output duration file.")
    parser.add_argument("--sample-rate", type=int, help="the sample of wavs.")
    parser.add_argument(
        "--n-shift",
        type=int,
        help="the n_shift of time_to_frames, also called hop_length.")
    parser.add_argument(
        "--config", type=str, help="config file with fs and n_shift.")

    args = parser.parse_args()
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    inputdir = Path(args.inputdir).expanduser()
    output = Path(args.output).expanduser()
    gen_duration_for_svs(inputdir, output, config.fs, config.n_shift)


if __name__ == "__main__":
    main()
