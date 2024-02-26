
This directory contains the evaluation script and data to run NASA-specific QA tasks in BEIR retrieval format. It has the following Evaluation Data sets:
1. NASA QA: QA set created from ~500 questions annotated by NASA SMEs (400 questions used in the test set, rest for validation)
2. PubMedQA: QA task from the BLURB benchmark for biomedical domain
3. BioASQ: QA task from the BLURB benchmark for biomedical domain


All datasets have been saved in the BEIR format, with queries depicting the answerable questions from each set, and corpus comprising of all the passages in the dataset along with 200K abstracts from the NASA corpus as confounding examples.
You can download the data in MTEB format (including the NASA corpus) from [here](https://drive.google.com/drive/folders/1uoyzV8Uur0p439FTJuZwIo1Q1qTzn292?usp=sharing)

To Evaluate the NASA-specific QA tasks (NASA-QA, PubMedQA, BioASQ) in a retrieval format, first install the mteb and beir libraries:
> pip install mteb beir

Then run
> python eval_qa_beir_for_nasa.py --model_name_or_path $MODEL_PATH --output_dir $OUTPUT_DIR 

--model_name_or_path must point to a Sentence Transformer Model. The results will be stored in JSON format in --output_dir. There will be one JSON file per dataset, with all metrics. We usually look at the `NDCG@10` metric. 

Please ensure the data is placed in the same directory as the script. Optionally, you can change the evaluation classes in `eval_qa_beir_for_nasa.py` such that `self.data_path` to point to your data copy for each class. Please refer to the [MTEB repo](https://github.com/embeddings-benchmark/mteb) for more info on creating custom evaluation tasks.
