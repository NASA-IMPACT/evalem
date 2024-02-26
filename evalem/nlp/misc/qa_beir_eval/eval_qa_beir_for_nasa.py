import logging
import argparse
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from beir.datasets.data_loader import GenericDataLoader
from datasets import DatasetDict

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

parser = argparse.ArgumentParser(description="MTEB Evaluation")
parser.add_argument("-m", "--model_name_or_path", help="path to sentence transformer model",
                    type=str, required=True)
parser.add_argument("-d", "--output_dir", help="path to output dir",
                    type=str, required=True)
parser.add_argument("--run_validation", help="whether to run validation for NASA QA",
                    action='store_true')

args = parser.parse_args()
print(args)


class NASAQA(AbsTaskRetrieval):

    @property
    def description(self):
        return {
            "name": self.name,
            "type": "Retrieval",
            "location": self.data_path,
            "category": "s2p",
            "eval_splits": ["test", "dev"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
    
    
    def __init__(self, **kwargs):
        self.name = "NASAQA"
        self.data_path = "/dccstor/aashka1/sentence_embeddings/data/sent_embed_data/nasa_qa_beir"
        super().__init__(**kwargs)
    
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        
        corpus_dict = {}
        query_dict = {}
        relevant_docs_dict = {}
        for split in self.description['eval_splits']:
            corpus, queries, relevant_docs = GenericDataLoader(data_folder=self.data_path).load(split=split)
    
            corpus_dict[split] = corpus
            query_dict[split] = queries
            relevant_docs_dict[split] = relevant_docs
        
        self.corpus = DatasetDict(corpus_dict)
        self.queries = DatasetDict(query_dict)
        self.relevant_docs = DatasetDict(relevant_docs_dict)
        # corpus, queries, relevant_docs = GenericDataLoader(data_folder=self.data_path).load(split="test")
        
        # eval_split = self.description['eval_splits'][0]
        # self.corpus = DatasetDict({eval_split:corpus})
        # self.queries = DatasetDict({eval_split:queries})
        # self.relevant_docs = DatasetDict({eval_split:relevant_docs})
        
        self.data_loaded = True

class NASAQA_EN(AbsTaskRetrieval):

    @property
    def description(self):
        return {
            "name": self.name,
            "type": "Retrieval",
            "location": self.data_path,
            "category": "s2p",
            "eval_splits": ["test", "dev"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
    
    
    def __init__(self, **kwargs):
        self.name = "NASAQA_EN"
        self.data_path = "nasa_qa_beir_enonly"
        super().__init__(**kwargs)
    
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        
        corpus_dict = {}
        query_dict = {}
        relevant_docs_dict = {}
        for split in self.description['eval_splits']:
            corpus, queries, relevant_docs = GenericDataLoader(data_folder=self.data_path).load(split=split)
    
            corpus_dict[split] = corpus
            query_dict[split] = queries
            relevant_docs_dict[split] = relevant_docs
        
        self.corpus = DatasetDict(corpus_dict)
        self.queries = DatasetDict(query_dict)
        self.relevant_docs = DatasetDict(relevant_docs_dict)
        
        self.data_loaded = True

class PubMedQA(AbsTaskRetrieval):

    @property
    def description(self):
        return {
            "name": self.name,
            "type": "Retrieval",
            "location": self.data_path,
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
    
    
    def __init__(self, **kwargs):
        self.name = "PubMedQA"
        self.data_path = "pubmedqa_beir"
        super().__init__(**kwargs)
    
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        
        corpus_dict = {}
        query_dict = {}
        relevant_docs_dict = {}
        for split in self.description['eval_splits']:
            corpus, queries, relevant_docs = GenericDataLoader(data_folder=self.data_path).load(split=split)
    
            corpus_dict[split] = corpus
            query_dict[split] = queries
            relevant_docs_dict[split] = relevant_docs
        
        self.corpus = DatasetDict(corpus_dict)
        self.queries = DatasetDict(query_dict)
        self.relevant_docs = DatasetDict(relevant_docs_dict)
        
        self.data_loaded = True

class BioASQ(AbsTaskRetrieval):

    @property
    def description(self):
        return {
            "name": self.name,
            "type": "Retrieval",
            "location": self.data_path,
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
    
    
    def __init__(self, **kwargs):
        self.name = "BioASQ"
        self.data_path = "bioasq_beir"
        super().__init__(**kwargs)
    
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        
        corpus_dict = {}
        query_dict = {}
        relevant_docs_dict = {}
        for split in self.description['eval_splits']:
            corpus, queries, relevant_docs = GenericDataLoader(data_folder=self.data_path).load(split=split)
    
            corpus_dict[split] = corpus
            query_dict[split] = queries
            relevant_docs_dict[split] = relevant_docs
        
        self.corpus = DatasetDict(corpus_dict)
        self.queries = DatasetDict(query_dict)
        self.relevant_docs = DatasetDict(relevant_docs_dict)
        
        self.data_loaded = True
    

tasks = ["NASAQA_EN", "PubMedQA", "BioASQ"]

model = SentenceTransformer(args.model_name_or_path)

logger.info(f"Evaluating tasks: {tasks}")

for task in tasks:
    logger.info(f"Running task: {task}")
    if task == 'NASAQA':
        eval_splits = ["test"] if not args.run_validation else ["dev", "test"]
    else:
        eval_splits = ["test"]
    evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
    evaluation.run(model, output_folder=f"{args.output_dir}", eval_splits=eval_splits)