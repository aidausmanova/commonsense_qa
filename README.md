# commonsense_qa

This project explores T5 Large Language Model (https://huggingface.co/transformers/v2.10.0/model_doc/t5.html).
The aim of this project is to report the training time and efficiency of the model. This is achieved through infusing external knowledge from ConceptNet Knowledge Graph and fine-tuning the model on the Commonsense Question Answering task. Training time, power consumption and approximate carbon emissions are tracked throughout all training processes via CarbonTracker (https://github.com/lfwa/carbontracker).

### ConceptNet
You can download ConceptNet assertions (https://github.com/commonsense/conceptnet5/wiki/Downloads) and save in "data/" folder.
To verbalize the ConcpetNet graph run the src/core/util/preprocess_conceptnet.py script.

### Knowledge Infusion
The knowlegde infusion step is done following the example of T5 (https://huggingface.co/docs/transformers/model_doc/t5#training) on Masked Language Modeling (MLM) using previously pre-processed ConceptNet triples.
To execute the task run src/core/new_mlm.py script.

### Fine-tuning
In the end, the T5 model is fine-tuned on the TellMeWhy dataset (https://stonybrooknlp.github.io/tellmewhy/) for the Commonsense QA task.
To execute fine-tuning run src/core/finetune_hf.py script.
