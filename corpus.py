# core modules
from os import listdir
from os.path import isfile, join

# extra modules
from conllu import parse_incr

# own modules
from features import extract_features


def _is_dataset_key_valid(dataset_key):
    valid_dataset_keys = ["dev", "train", "test"]
    if dataset_key in valid_dataset_keys:
        return dataset_key

    raise ValueError(f"Please choose one of {valid_dataset_keys}")


def parse_from_conllu(pathname_dir, lang_codes, dataset_key="train"):
    dataset_key = _is_dataset_key_valid(dataset_key)

    corpus = []
    features = []

    for lang_idx, lang_code in enumerate(lang_codes):
        print(f"Parsing {lang_code}... ", end="")
        lang_dir = f"{pathname_dir}/{lang_code}"
        filenames = [join(lang_dir, f) for f in listdir(lang_dir) if isfile(join(lang_dir, f))]

        for filename in filenames:
            if dataset_key in filename:
                with open(filename, "r", encoding="utf-8") as f:
                    corpus.append(list(parse_incr(f)))
                    print("extracting features... ", end="")
                    features.append(extract_features(corpus[lang_idx]))
        print("done !")

    return corpus, features


if __name__ == "__main__":
    import pandas as pd
    base_path = "/home/hippo/GoogleDrive/univ/master-2/TAL/projet/data"

    corpus_path = base_path + "/corpus"
    scores_path = base_path + "/scores.txt"

    scores = pd.read_csv(scores_path, sep=",", header=None)
    scores.columns = ['lang', 'las', 'uas']

    lang_codes = scores['lang']
    las = scores['las']
    uas = scores['uas']

    # ---
    # test with all languages
    # ---
    corpus, features = parse_from_conllu(pathname_dir=corpus_path,
                                         lang_codes=lang_codes)

    phrases = corpus[0]
    phrase = phrases[0]
    mot = phrase[0]

    print(f"lang_code\n\t{lang_codes[0]}")
    print(f"première phrase\n\t{phrase}")
    print(f"premier mot\n\t{mot}")
    print(f"nombre de clés disponibles par mot\n\t{len(mot)}")
    print(f"clés disponibles par mot\n\t{mot.keys()}")

    print(f"features\n\t{features[0]}")
    print(f"scores\n\tlas={las[0]}, uas={uas[0]}")
