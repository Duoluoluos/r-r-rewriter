from retriever_tools.doc_retriever import SyllabusDocRetriever, LIRetriever
from retriever_tools.llm_rewriter import LLM_Rewriter
from datetime import datetime
import os
import json
from tqdm import tqdm
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import logging
import pandas as pd
import retriever_tools.utils as utils
from retriever_tools.llm_reranker import DocumentRanker
import numpy as np
from retriever_tools.web_search import search_web 


class LangchainHandler:
    def __init__(self, llm_type, **kwargs):
        self.llm = self.create_llm(llm_type, **kwargs)
        self.read_chain = self.create_read_chain()
    
    @staticmethod
    def create_llm(llm_type, **kwargs):
        if llm_type == 'o1-preview':
            return ChatOpenAI(model_name='o1-preview', 
                              temperature=kwargs["temperature"],
                              top_p=kwargs["top_p"])
        elif 'gpt' in llm_type:
            return ChatOpenAI(model_name=llm_type, temperature=0)
        else:
            raise ValueError("Unsupported LLM type")

    def build_prompt(self, type):
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer these quetions based on the web search results."),
            ("human", "Here is the question:\n{question}\n web search results:\n{context}"),
        ])


    def create_read_chain(self):
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.build_prompt(type='reader')
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


class ExamValidator:
    def __init__(self, data_path):
        #self.rewriter = LLM_Rewriter(prlm_path, use_langchain=config["use_langchain"])
        self.llm_reader = LangchainHandler(langchain_reader_llm, temperature=0, top_p=1.0)
        with open(data_path, "r") as f:
            self.data = json.load(f)
        if "data" in self.data:
            self.data = self.data["data"]
        if "answers" in self.data[0]:
            self.data = [{"id": d["id"], "question": d["question"], "answer": d["answers"]} for d in self.data]
        for i, d in enumerate(self.data):
            answers = []
            for annotation in d["annotations"]:
                assert annotation["type"] in ["singleAnswer", "multipleQAs"]
                if annotation["type"]=="singleAnswer":
                    answers.append([list(set(annotation["answer"]))])
                else:
                    answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers for _answer in answer for _a in _answer])
            self.data[i]["answer"] = answers

    def case_study(self):
        d = self.data[1984]
        question = d["question"]
        rewrited_question = llm_rewriter.RaFe(question, use_langchain=config["use_langchain"], model_type=langchain_rewriter_llm, temperature=0)
        print(question)
        print(rewrited_question)
        answers = d["answer"]
        id = d["id"]
        searched_docs = search_web(rewrited_question, k=4)
        searched_doc =  "\n\n".join([doc['text'] for doc in searched_docs])
        res = self.llm_reader.read_chain.invoke({"question": rewrited_question, "context": searched_doc})

    @staticmethod
    def postprocess_query(query):
        if '*' in query:
            query = query.replace("*", "")
        if ":" in query:
            query = query.split(":")[-1]
        if "\"" in query:
            query = query.replace("\"", "")   
        return query

    def validate(self, test_type = 'rr'):
        # 使用load_reference函数读取json文件
        prediction = {}
        count  = 0
        for d in tqdm(self.data):
            # count += 1
            # if count > 5:
            #     break
            question = d["question"]
            rewritten_question = llm_rewriter.RR_Rewrite(question, doc_label='AmbigQA' ,use_langchain=config["use_langchain"], model_type=langchain_rewriter_llm, temperature=0)
            rewritten_question = self.postprocess_query(rewritten_question)
            # print(rewritten_question)
            answers = d["answer"]
            id = d["id"]
            searched_docs = search_web(rewritten_question, k=4)
            searched_doc =  "\n\n".join([doc['text'] for doc in searched_docs])
            res = self.llm_reader.read_chain.invoke({"question": rewritten_question, "context": searched_doc})
            prediction[id] = [res]
        self.save_predictions(prediction, test_type)

    def save_predictions(self, predictions, test_type):
        save_path = f"./log/ambigqa_predictions_{test_type}.json"
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        print("Saved prediction in {}".format(save_path))


if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    langchain_rewriter_llm = config["langchain_rewriter_type"]
    langchain_reader_llm = config["langchain_reader_type"]
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    # Configuration
    os.environ["TOKENIZERS_PARALLELISM"]="True"
    PrLM_path = config["pretrained_rewriter_model"]
    llm_rewriter = LLM_Rewriter(PrLM_path, use_langchain=config["use_langchain"])
    if config["use_langchain"] is False:
        print(f"Using Pretrained Rewriter model: {PrLM_path}")
    exp_db = None
    validator = ExamValidator(data_path="/data/wangqi/AmbigQA/dev_with_evidence_articles.json")
    #validator.case_study()
    validator.validate()