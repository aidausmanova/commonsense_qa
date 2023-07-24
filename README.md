# commonsense_qa

This project was created in the scope of the program at the University of Hamburg.

### ConceptNet
You can download ConceptNet assertions (https://github.com/commonsense/conceptnet5/wiki/Downloads) and save in "data/" folder.
To verbalize the ConcpetNet graph run the src/core/util/preprocess_conceptnet.py script.

### Pre-training
The pre-training is done following the example of T5 (https://huggingface.co/docs/transformers/model_doc/t5#training) on Masked Language Modeling (MLM) using previously pre-processed ConceptNet triples.
To execute the pre-training task run src/core/new_mlm.py script.

### Fine-tuning
In the end, the T5 model is fine-tuned on the TellMeWhy dataset (https://stonybrooknlp.github.io/tellmewhy/) for the Commonsense QA task.
To execute fine-tuning run src/core/finetune_hf.py script.
