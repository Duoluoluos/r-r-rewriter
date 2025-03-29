from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
from langchain_community.docstore.base import Docstore
from langchain.docstore.document import Document
import pickle
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import sys
sys.path.append('.')
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retriever_tools.utils import str_parser

class DocumentRanker:
    def __init__(self, model_name, device = 'cuda', use_langchain=False):
        self.use_langchain = use_langchain
        if use_langchain == False:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
            self.device = device

    def hf_rerank(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt", padding="longest", max_length=4096, truncation=True, 
                                        return_attention_mask=True, add_special_tokens=True).to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            attention_mask=model_inputs['attention_mask'].to(self.device), do_sample=True, temperature=0.1, 
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def langchain_rerank(self, query):
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt =[("system", "You are a helful assistant"),
                ("human", f"{query}"),]
        llm_chain = llm | StrOutputParser
        response = llm.invoke(prompt)
        return response.content

    def rank(self, query, document):
        # Reproduction for RaFe
        prompt = f"""Query: {query}\n\
        Document: {document}\n\
        Score the relevance of the document to the query.\n\
        The relevance score is based on how relevant you think the document is to the question, with scores ranging from 1 to 10. Do not include any documents that are not relevant to the question.\n\
        Please represent your score in json dictionary format, with the key being the string "score" and the value being the score. For example: {{"score":10}}"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        if self.use_langchain is False:
            score = self.hf_rerank(messages)
        else:
            score = self.langchain_rerank(prompt)
        score = str_parser(score)
        try:
            score_num = json.loads(score)["score"]
            return score_num
        except:
            return score

class QueryProcessor:
    def __init__(self, retriever, ranker):
        self.retriever = retriever
        self.ranker = ranker

    def process_query(self, query, k=3):
        retrieved_docs = self.retriever.retrieve(query, k)
        ranked_docs = self.ranker.rank(query, retrieved_docs)
        return ranked_docs

if __name__ == '__main__':
    pass
