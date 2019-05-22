import os
from enum import Enum, unique

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    # directory for each main directory
    CORPUS_DIR = os.path.join(basedir, "data")
    SENSITIVE_KEYS_PATH = os.path.join(CORPUS_DIR, "keyword.csv")
    SENSITIVE_RULES_PATH = os.path.join(CORPUS_DIR, "rules.txt")
    STOP_WORDS_PATH = os.path.join(CORPUS_DIR, "stop_words.utf8")
    NODE_CHILD_NAME = "c"
    NODE_END_NAME = "e"
    NODE_TOKEN_NAME = "t"


@unique
class LabelType(Enum):
    Terror = "terror"
    Ads = "ads"
    Normal = "normal"
    Pulp = "pulp"
    Politician = "politician"
    Other = "other"
