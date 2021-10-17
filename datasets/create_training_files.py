import dataclasses
import json
import os
import pathlib
import random
import shutil
import sys
import unicodedata as ud
from shutil import rmtree

import pydub
from progressbar import progressbar
from pydub import AudioSegment


@dataclasses.dataclass
class CreateConfig:
    voices: list[str] = dataclasses.field(default_factory=list)
    languages: list[str] = dataclasses.field(default_factory=list)

    def load(self, config_file: str):
        config: dict
        if not os.path.exists(config_file):
            self.save(config_file)
            return
        try:
            with open(config_file, "r") as r:
                config = json.load(r)
        except json.JSONDecodeError:
            self.save(config_file)
            return
        if type(config) == dict:
            for k, v in config.items():
                setattr(self, k, v)
        else:
            self.save(config_file)

    def save(self, config_file: str):
        with open(config_file, "w") as w:
            json.dump(dataclasses.asdict(self), w, indent=4)


def main():
    argv0: str = sys.argv[0]
    if argv0:
        workdir: str = os.path.dirname(argv0)
        if workdir:
            os.chdir(workdir)
    workdir = os.getcwd()

    cfg: CreateConfig = CreateConfig()
    cfg.load("create_config.json")

    punctuations_out = '、。，"(),.:;¿?¡!\\'
    punctuations_in = '\'-'

    for file in ["train.txt", "val.txt", "all.txt"]:
        if os.path.exists(file):
            os.remove(file)

    for _ in ["lin_spectrograms", "mel_spectrograms"]:
        rmtree(os.path.join(workdir, _), ignore_errors=True)

    for parent in [  #
            "cherokee-audio-data-private/beginning-cherokee",  #
            "cherokee-audio-data-private/durbin-feeling",  #
            "cherokee-audio-data-private/thirteen-moons-disk1",  #
            "cherokee-audio-data-private/thirteen-moons-disk2",  #
            "cherokee-audio-data-private/thirteen-moons-disk3",  #
            "cherokee-audio-data-private/thirteen-moons-disk4",  #
            "cherokee-audio-data-private/thirteen-moons-disk5",  #

            "cherokee-audio-data/durbin-feeling-tones",  #
            "cherokee-audio-data/michael-conrad2",  #
            "cherokee-audio-data-private/sam-hider",  #
            "cherokee-audio-data/see-say-write",  #
            "cherokee-audio-data/cno",  #
            "cherokee-audio-data/walc-1",  #
            "cherokee-audio-data/wwacc",  #
    ]:
        for txt in ["all.txt", "val.txt", "train.txt"]:
            with open(pathlib.Path(parent).joinpath(txt), "r") as f:
                lines: list = []
                for line in f:
                    line = ud.normalize("NFC", line.strip())
                    line = line.replace("|wav/", "|" + parent + "/wav/")
                    lines.append(line)

                random.Random(len(lines)).shuffle(lines)
                with open(txt, "a") as t:
                    for line in lines:
                        t.write(line)
                        t.write("\n")

    # get char listing needed for params file
    letters = ""
    chars: list = []
    with open("all.txt", "r") as f:
        for line in f:
            fields = line.split("|")
            text: str = fields[6].lower()
            text = ud.normalize("NFC", text)
            for c in text:
                if c in punctuations_in or c in punctuations_out:
                    continue
                if c in chars:
                    continue
                chars.append(c)
        chars.sort()
        for c in chars:
            letters += str(c)
        config: dict = dict()
        config["characters"] = letters
        tmp = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=3)
        with open("json-characters.json", "w") as j:
            j.write(tmp)
            j.write("\n")

    # rewrite shuffled
    lines = []
    with open("train.txt", "r") as t:
        for line in t:
            lines.append(line)
    random.Random(len(lines)).shuffle(lines)
    with open("train.txt", "w") as t:
        for line in lines:
            t.write(line)

    # rewrite shuffled
    lines = []
    with open("val.txt", "r") as v:
        for line in v:
            lines.append(line)
    random.Random(len(lines)).shuffle(lines)
    with open("val.txt", "w") as v:
        for line in lines:
            v.write(line)

    # create train.json
    all_lines: list[str] = lines.copy()
    with open("all.txt", "r") as f:
        for line in f:
            all_lines.append(line.strip())
    train_entries: list[tuple[str, str, str]] = list()
    line: str
    for line in all_lines:
        fields = line.split("|")
        record_id: str = fields[0]
        voice: str = fields[1]
        lang: str = fields[2]
        wav: str = fields[3]
        text: str = fields[6]
        if voice not in cfg.voices or lang not in cfg.languages:
            continue
        train_entries.append((wav, record_id, text))

    print(f"Preparing {len(train_entries):,} wavs for preprocessing.")

    shutil.rmtree("wavs", ignore_errors=True)
    os.mkdir("wavs")
    bar = progressbar.ProgressBar(maxval=len(train_entries))
    bar.start()

    train_list: list[tuple[str, str]] = list()
    for wav, record_id, text in train_entries:
        audio: AudioSegment = AudioSegment.from_file(wav)
        audio = audio.set_channels(1).set_frame_rate(16000)
        output_wav_path = pathlib.Path("wavs", pathlib.Path(wav).stem).with_suffix(".wav")
        audio.export(output_wav_path, format="wav")
        wav_json_entry: str = str(pathlib.Path("wavs") / output_wav_path.stem)
        npy_json_entry: str = str(pathlib.Path("train") / output_wav_path.stem)
        train_list.append((wav_json_entry, npy_json_entry))
        bar.update(bar.currval + 1)
    with open("train.json", "w") as w:
        json.dump(train_list, w, indent=1)

    # key = path.stem
    # key: transcript for key, _, transcript in text if key in train_set
    with open("transcripts.txt", "w") as w:
        for wav, record_id, text in train_entries:
            key = pathlib.Path(wav).stem
            w.write(f"{key}|{record_id}|{text}")
            w.write("\n")
    bar.finish()

    sys.exit(0)


if __name__ == "__main__":
    # c: CreateConfig = CreateConfig()
    # c.load("create_config.json")
    main()
