from retriever_tools.doc_retriever import SyllabusDocRetriever, LIRetriever
from retriever_tools.llm_rewriter import LLM_Rewriter
from retriever_tools.utils import extract_number
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
import numpy as np
import textwrap

class LangchainHandler:
    def __init__(self, llm_type, exp_db, textbook_db, **kwargs):
        self.llm = self.create_llm(llm_type, **kwargs)
        if exp_db:
            self.exp_retriever = exp_db.as_retriever()
        if textbook_db:
            self.tb_retriever = textbook_db.as_retriever()
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

    def build_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", "You are a teaching assistant. Answer questions from students on course logistics using the attached course syllabus document in your knowledge base. If an answer is not contained in the course syllabus output ‘No/insufficient information’"),
            ("human", "Here is the question you need to answer, along with some advice for tackling it:\n\n{question}\n\nHere is the syllabus document that provides additional context:\n\n{context}"),
        ])


    def create_read_chain(self):
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.build_prompt()
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
    def calculate_score(query: str, prediction: str, ground_truth: str):
        prompt = f'''
            Your job is to evaluate the similarity of different answers to a single question. 
            You will be given a question asking for information regarding a specific college course.
            You will also be given two possible answers to that question, and will have to evaluate the claims in one answer against the other. 

            Steps: 
            1. List all of the atomic claims made by Answer 1. Note that an answer saying that there is no information counts as a single claim.
            2. Tell me which of those claims are supported by Answer 2. 
            3. Summarize the results using the template "Score: <num supported claims>/<num total claims>". 
            Ensure that both numbers are integers.
            Question: {query}
            Answer 1: {prediction}
            Answer 2: {ground_truth}.
        '''
        prompt = textwrap.dedent(prompt)
        llm = ChatOpenAI(model_name="gpt-4o",
            temperature=0,
            top_p=1.0,
            n=1)
        return llm.invoke(prompt).content.strip()


    def case_study(self, query):
        try:
            rewrited_query = self.rewriter.RR_Rewrite(query, doc_label='SyllabusQA', use_langchain=config["use_langchain"], model_type=langchain_rewriter_llm)
            docs = self.dpr_chain.invoke(rewrited_query)
            for doc in docs:
                print(doc.page_content)
                print('*****************************')
        except Exception as e:
            print(e)


    def validate(self, exam_fn, k=3):
        log_dic = dict()
        df = pd.read_csv(exam_fn)
        precision_list = []
        recall_list = []
        f1_list = []
        for i in tqdm(range(1, len(df))):
        # for i in tqdm(range(1,31)):
            try:    
                sub_df = df.iloc[i]
                query = sub_df['question']
                ground_truth = sub_df['answer']
                syllabi = sub_df['syllabus_name']
                rewrited_query = self.rewriter.RR_Rewrite(query, doc_label='SyllabusQA', use_langchain=config["use_langchain"], model_type=langchain_rewriter_llm)
                multi_queries = retriever_utils.extract_intent(rewrited_query)[0]
                # print(query)
                # print(rewrited_query)
                retrieved_doc = ""
                docs = self.dpr_chain.invoke(query + '\n'  + multi_queries[0] , syllabi, k)
                # print(f"Retrieved Doc: {docs}")
                for t in range(k):               
                    retrieved_doc += docs[t].page_content + "\n\n"
                res = self.read_chain.invoke({"question": query, "context": retrieved_doc})
                precision = self.calculate_score(query, res, ground_truth)
                precision = precision.split(':')[-1].strip()
                precision = eval(precision)
                if precision > 1:
                    precision = 1/precision
                recall = self.calculate_score(query, ground_truth, res)
                recall = recall.split(':')[-1].strip()
                recall = eval(recall)
                if recall > 1:
                    recall = 1/recall
                #print(f"Res: {res}\nGroundtruth:{ground_truth}\nPrecision: {precision}, Recall: {recall}")
                precision_list.append(precision)
                recall_list.append(recall)
            except:
                continue
                #raise Exception(f"Error: {query}")
        print(f"Precision: {sum(precision_list)/len(precision_list)}, Recall: {sum(recall_list)/len(recall_list)}")
        log_dic['precision'] = sum(precision_list)/len(precision_list)
        log_dic['recall'] = sum(recall_list)/len(recall_list)
        log_dic['f1'] = utils.get_f1_score(log_dic['precision'], log_dic['recall'])
        print(f"F1: {log_dic['f1']}")



if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    langchain_rewriter_llm = config["langchain_rewriter_type"]
    langchain_reader_llm = config["langchain_reader_type"]
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    # Configuration
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    doc_path = 'data/SyllabusQA/syllabi/syllabi_redacted/text'
    PrLM_path = config["pretrained_rewriter_model"]
    if config["use_langchain"] is False:
        print(f"Using Pretrained Rewriter model: {PrLM_path}")
    exp_db = None
    # textbook = SyllabusDocRetriever(doc_folder=doc_path)
    base_retriever = SyllabusDocRetriever(doc_path, use_embeddings=True)
    retriever_utils = LIRetriever(retriever = base_retriever, k=3)
    # Create model handler and RAG chain
    model_handler = LangchainHandler(langchain_reader_llm, None, None, **config["model_kwargs"])
    # Validate exams
    validator = ExamValidator(dpr_chain = base_retriever, read_chain = model_handler.read_chain, prlm_path = PrLM_path)
    validator.validate(exam_fn="data/SyllabusQA/data/dataset_split/val.csv")
    validator.case_study(query="What is the name of the professor who teaches this class?")
