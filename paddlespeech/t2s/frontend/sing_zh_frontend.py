# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Dict
from typing import List

import librosa
import numpy as np
import paddle

from paddlespeech.t2s.datasets.get_feats import Pitch
from paddlespeech.t2s.frontend.zh_frontend import Frontend


class SingFrontend():
    def __init__(self,
                 g2p_model="pypinyin",
                 phone_vocab_path=None,
                 tone_vocab_path=None):

        self.zh_frontend = Frontend(
            phone_vocab_path=phone_vocab_path, tone_vocab_path=tone_vocab_path)
        if 'SP' in self.zh_frontend.vocab_phones.keys():
            self.zh_frontend.vocab_phones['sp'] = self.zh_frontend.vocab_phones[
                'SP']
        self.sample_rate = None
        self.hop_length = None

    def parse_lyrics(self, sentence: str):
        lyrics = sentence.split("<type:lyrics>")[1].split("<type:")[0]
        chars = []
        zh_chars = []
        pinyin_chars = []
        pre_char = ''
        for char in lyrics:
            if char >= '\u4e00' and char <= '\u9fa5':
                if pre_char != '':
                    chars += [pre_char]
                    zh_chars += [None]
                    pinyin_chars += [None]
                    pre_char = ''
                chars += [char]
                zh_chars += [char]
                pinyin_chars += [None]
            elif char == ' ':
                if pre_char != '':
                    chars += [pre_char]
                    zh_chars += [None]
                    pinyin_chars += [None]
                pre_char = ''
            else:
                pre_char += char
                if char == '>' and pre_char.startswith('<pinyin:'):
                    pinyin = pre_char.split('<pinyin:')[1].split('>')[0]
                    chars += [None]
                    zh_chars += [None]
                    pinyin_chars += [pinyin]
                    pre_char = ''

        if pre_char.startswith('<pinyin:') and pre_char.endswith('>'):
            pinyin = pre_char.split('<pinyin:')[1].split('>')[0]
            chars += [None]
            zh_chars += [None]
            pinyin_chars += [pinyin]
            pre_char = ''
        if pre_char != '':
            chars += [pre_char]
            zh_chars += [None]
            pinyin_chars += [None]

        zh_sentence = ''.join([char for char in zh_chars if char is not None])
        phonemes = self.zh_frontend.get_phonemes(
            sentence=zh_sentence, merge_sentences=False)[0]
        nest_phonemes = []
        pre_phoneme = ''
        for phoneme in phonemes:
            phoneme = phoneme.rstrip('1234567890').replace('iii', 'i').replace(
                'ii', 'i')
            if phoneme == 'i' and (pre_phoneme[:1] in 'aeiouv' or
                                   pre_phoneme == ''):
                pre_phoneme = 'y'
            if phoneme[:1] == 'u' or phoneme[:1] == 'v':
                if pre_phoneme[:1] in 'aeiouv' or pre_phoneme == '':
                    pre_phoneme = 'w' if phoneme[:1] == 'u' else 'y'
                    phoneme = phoneme[1:]
                elif len(phoneme) > 2:
                    phoneme = phoneme[0] + phoneme[2:]
            if phoneme == 'iou':
                phoneme = 'iu'
            if phoneme[:1] in 'aeiouv':
                if pre_phoneme[:1] not in 'aeiouv':
                    nest_phonemes.append([pre_phoneme, phoneme])
                else:
                    nest_phonemes.append([phoneme])
            pre_phoneme = phoneme

        for i in range(len(zh_chars)):
            char = chars[i]
            zh_char = zh_chars[i]
            pinyin = pinyin_chars[i]
            if zh_char is None:
                if char is not None:
                    nest_phonemes.insert(i, [char])
                elif pinyin is not None:
                    if len(pinyin) > 2 and pinyin[1:2] == 'h':
                        pinyin = [pinyin[:2], pinyin[2:]]
                    elif pinyin[:1] == 'aeiouv':
                        pinyin = [pinyin]
                    else:
                        pinyin = [pinyin[:1], pinyin[1:]]
                    nest_phonemes.insert(i, pinyin)

        return nest_phonemes

    def parse_note(self, sentence: str, nest_phonemes: List[List[str]]):
        notes = sentence.split("<type:note>")[1].split("<type:")[0]
        notes = [note.split(' ') for note in notes.split('|')]
        nest_notes = [[n for n in note if n != ''] for note in notes]
        for notes, phonemes in zip(nest_notes, nest_phonemes):
            inserted_phones = 0
            if len(notes) > 1:
                inserted_phones = 1
                phonemes.append('<slur>')
            if len(phonemes) > 1 + inserted_phones:
                notes.insert(0, notes[0])
        notes = [note for notes in nest_notes for note in notes]
        pitch = Pitch().get_pitch_by_note(notes).reshape(-1, 1)

        return np.float32(np.asarray(pitch)), nest_phonemes

    def parse_duration(self, sentence: str, nest_phonemes: List[List[str]]):
        durations = sentence.split("<type:duration>")[1].split("<type:")[0]
        durations = [duration.split(' ') for duration in durations.split('|')]
        nest_durations = [[float(n) for n in duration if n != '']
                          for duration in durations]
        for durations, phonemes in zip(nest_durations, nest_phonemes):
            inserted_phones = 0
            if '<slur>' in phonemes:
                inserted_phones = 1
            if len(phonemes) > 1 + inserted_phones:
                durations.insert(0, 0.04)
                durations[1] -= 0.04
        durations = [
            duration for durations in nest_durations for duration in durations
        ]
        ends = []
        end = 0
        for duration in durations:
            end += duration
            ends.append(end)
        frame_pos = librosa.time_to_frames(
            ends, sr=self.sample_rate, hop_length=self.hop_length)
        durations = np.diff(frame_pos, prepend=0)

        return np.int64(np.asarray(durations))

    def get_input_ids(self,
                      sentence: str,
                      merge_sentences: bool=False,
                      get_tone_ids: bool=False,
                      add_sp: bool=True,
                      to_tensor: bool=True) -> Dict[str, List[paddle.Tensor]]:

        sentences = sentence.replace('\r\n', '\n').split('\n')
        phone_ids_list = []
        durations_list = []
        pitch_list = []
        for sentence in sentences:
            nest_phonemes = self.parse_lyrics(sentence)
            pitch, nest_phonemes = self.parse_note(sentence, nest_phonemes)
            durations = self.parse_duration(sentence, nest_phonemes)
            phonemes = [
                phoneme for phonemes in nest_phonemes for phoneme in phonemes
            ]
            phone_ids = self.zh_frontend._p2id(phonemes)
            phone_ids_list.append(phone_ids)
            durations_list.append(durations)
            pitch_list.append(pitch)

        if merge_sentences:
            phone_ids_list = np.concatenate(phone_ids_list, 0)
            durations_list = np.concatenate(durations_list, 0)
            pitch_list = np.concatenate(pitch_list, 0)[..., None]
            if to_tensor:
                phone_ids_list = paddle.to_tensor(phone_ids_list)
                durations_list = paddle.to_tensor(durations_list)
                pitch_list = paddle.to_tensor(pitch_list)
        elif to_tensor:
            phone_ids_list = [
                paddle.to_tensor(phone_ids) for phone_ids in phone_ids_list
            ]
            durations_list = [
                paddle.to_tensor(durations) for durations in durations_list
            ]
            pitch_list = [paddle.to_tensor(pitch) for pitch in pitch_list]

        result = {
            'phone_ids': phone_ids_list,
            'durations': durations_list,
            'pitch': pitch_list
        }

        return result
