from retriever_tools.doc_retriever import FintextQARetriever, LIRetriever
from retriever_tools.llm_rewriter import LLM_Rewriter
from datetime import datetime
import os
import json
import textwrap
from tqdm import tqdm
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import logging
import pandas as pd
import retriever_tools.utils as utils


class ModelHandler:
    def __init__(self, llm_type, exp_db, textbook_db, **kwargs):
        self.llm = self.create_llm(llm_type, **kwargs)
        if exp_db:
            self.exp_retriever = exp_db.as_retriever()
        if textbook_db:
            self.tb_retriever = textbook_db.as_retriever()
        self.rag_chain = self.create_rag_chain()
    
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
            ("system", "You are a finance assistant. You will be given a financial question and a document, and your task is to answer the question based on the document."),
            ("human", "Here is the question:\n{question}\n financial document:\n{fintext_doc}\n"),
        ])


    def create_rag_chain(self):
        return (
            {"question": RunnablePassthrough(), "fintext_doc": RunnablePassthrough()}
            | self.build_prompt(type='reader')
            | self.llm
            | StrOutputParser()
        )
    

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

class ExamValidator:
    def __init__(self, dpr_chain, read_chain, prlm_path):
        self.read_chain = read_chain
        self.current_date = datetime.now().strftime("%m%d")
        self.rewriter = LLM_Rewriter(prlm_path, use_langchain=config["use_langchain"])
        self.dpr_chain = dpr_chain

    @staticmethod
    def calculate_score(prediction: str, ground_truth: str):
        acc = utils.compute_acc(ground_truth, prediction)
        bleu = utils.compute_bleu(ground_truth, prediction)
        rouge = utils.compute_rouge(ground_truth, prediction)
        return acc, bleu, rouge

    def validate(self, exam_fn, k=1):
        log_dic = dict()
        df = pd.read_csv(exam_fn)
        acc_list, bleu_list, rouge_list = [], [], []
        for i in tqdm(range(1, len(df))):
            sub_df = df.iloc[i]
            query = sub_df['question']
            ground_truth = sub_df['answer']
            rewrited_query = self.rewriter.RR_Rewrite(query, doc_label='FintextQA', use_langchain=config["use_langchain"], model_type=langchain_rewriter_llm)
            retrieved_doc = ""
            docs = self.dpr_chain.retrieve(rewrited_query, preprocess=False)
            for t in range(k):               
                retrieved_doc += docs[t].page_content + "\n\n"
            try:
                prediction = self.read_chain.invoke({"question": query, "fintext_doc": retrieved_doc})
                acc, bleu, rouge = self.calculate_score(prediction, ground_truth)
                acc_list.append(acc)
                bleu_list.append(bleu)
                rouge_list.append(rouge)
            except:
                continue

        log_dic["acc"] = acc_list
        log_dic["bleu"] = bleu_list
        log_dic["rouge"] = rouge_list
        with open(f"log/fintext_{self.current_date}.json", "w") as f:
            json.dump(log_dic, f, indent=4)


if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    langchain_rewriter_llm = config["langchain_rewriter_type"]
    langchain_reader_llm = config["langchain_reader_type"]
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    # Configuration
    os.environ["TOKENIZERS_PARALLELISM"]="True"
    doc_path = 'data/FintextQA/doc'
    PrLM_path = config["pretrained_rewriter_model"]
    exp_db = None
    textbook_db = FintextQARetriever(doc_folder=doc_path).vectorstore
    base_retriever = FintextQARetriever(doc_path, use_embeddings=True).vectorstore.as_retriever(search_type="similarity")
    # 初始化LI-Retriever
    li_retriever = LIRetriever(base_retriever, k=1)
    # Create model handler and RAG chain
    model_handler = ModelHandler(langchain_reader_llm, exp_db, textbook_db, **config["model_kwargs"])
    # Validate exams
    validator = ExamValidator(dpr_chain = li_retriever, read_chain = model_handler.rag_chain,
                               prlm_path = PrLM_path)
    validator.validate(exam_fn="data/FintextQA/fin_dataset_test.json")


