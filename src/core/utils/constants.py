from os.path import dirname, join
from pathlib import Path

PRETRAIN_PARAMETERS = {
    "MODEL": 't5-small',
    "TOKENIZER": 't5-small',
    "LEARNING_RATE": 1e-3,
    "TRAIN_EPOCHS": 3,
    "VAL_EPOCHS": 1,
    "TRAIN_BATCH_SIZE": 10,
    "VALID_BATCH_SIZE": 1,
    "MAX_SOURCE_TEXT_LENGTH": 512,
    "MAX_TARGET_TEXT_LENGTH": 50,
    "SEED": 42
}

FINETUNE_PARAMETERS = {
    "MODEL": 't5-small',
    "TOKENIZER": 't5-small',
    "LEARNING_RATE": 0.0001,
    "TRAIN_EPOCHS": 3,
    "VAL_EPOCHS": 1,
    "TRAIN_BATCH_SIZE": 8,
    "VALID_BATCH_SIZE": 1,
    "MAX_SOURCE_TEXT_LENGTH": 396,
    "MAX_TARGET_TEXT_LENGTH": 32,
    "SEED": 42
}

PROJECT_ROOT_DIR = dirname(Path(__file__).resolve().parents[2])

RELATION_TEMPLATES = {
    "RelatedTo": "{0} is related to {1}",
    "FormOf": "{0} is a form of {1}",
    "IsA": "{0} is a {1}",
    "NotIsA": "{0} is not a {1}",
    "PartOf": "{0} is part of {1}",
    "HasA": "{0} has {1}",
    "UsedFor": "{0} is used for {1}",
    "CapableOf": "{0} can {1}",
    "AtLocation": "{0} is at {1}",
    "Causes": "{0} causes {1}",
    "HasSubevent": "Something you do while {0} is {1}",
    "HasFirstSubevent": "First thing you do when {0} is {1}",
    "HasLastSubevent": "Last thing you do when {0} is {1}",
    "HasPrerequisite": "{0} requires {1}",
    "HasProperty": "{0} is {1}",
    "MotivatedByGoal": "You would {0} because {1}",
    "ObstructedBy": "{0} can be prevented by {1}",
    "CreatedBy": "{0} is created by {1}",
    "Synonym": "{0} is like {1}",
    "Antonym": "{0} is opposite of {1}",
    "DistinctFrom": "{0} is different from {1}",
    "DerivedFrom": "Word {0} derived from word {1}",
    "SymbolOf": "{0} is a symbol of {1}",
    "DefinedAs": "{0} is defined as {1}",
    "Entails": "{1} happens when {0} happens",
    "MannerOf": "{0} is a specific form of {1}",
    "LocatedNear": "{0} is near {1}",
    "SimlarTo": "{0} is similar to {1}",
    "EtymologicallyRelatedTo": "{0} and {1} words have the same origin",
    "EtymologicallyDerivedFrom": "Word {0} comes from word {1}",
    "CausesDesire": "{0} makes people want {1}",
    "MadeOf": "{0} is made of {1}",
    "ReceivesAction": "{0} can be {1}"
}