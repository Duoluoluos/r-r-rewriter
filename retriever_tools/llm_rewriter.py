from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
import pickle
import sys
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
sys.path.append('.')
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retriever_tools.utils import hf_inference

# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


class LLM_Rewriter:
    def __init__(self, model_name, device = 'cuda', use_langchain = False):
        if use_langchain is False:
            self.model_path = model_name
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.instruction = {"en": "Please rewrite this query, providing a rewritten version, to overcome the limitations of vector-distance-based retrieval systems, thereby enhancing the system's retrieval capabilities.",
                                "cn": "请重写这个查询，提供一个重写版本，以克服向量距离检索系统的局限性，从而增强系统的检索能力。"}
    def HyDE(self, query, use_langchain,model_type = "gpt-4o",answer = None, doc_label = "保荐代理人考试"):
        if answer is None:
            if doc_label == "保荐代理人考试":
                prompt =  self.instruction['cn']  + '\n' +  f'''请结合这道{doc_label}试题，写一段与该试题相关的参考资料。\n\
                        参考样例：\n问题：D在资本化期间内，尚未动用的借款进行暂时性投资取得的投资收益，应冲减资本化金额\n\
                        参考资料：专门借款在资本化期间确认的利息收入应扣除所支付的利息费用(包括因占用其他专门借款而发生的利息支出)后的金额，即全部投资收益都应当资本化。\n| 事项 | 结论 |\n| --- | --- |\n| 专门借款在资本化期间确认的利息收入扣<br>除所支付的利息后，全额资本化 | ①专门借款在资本化期间确认的利息收入扣除了所支付的利息费用(包括占<br>用其他专门借款而发生的利息支出)之后的差额，应当资本化<br>②专门借款未能获得收益(例如取得的投资收益不足以弥补所发生的借款费<br>用)，专门借款在资本化期间确认的利息收入不足以抵扣借款费用的，差额部<br>分应当予以资本化<br>③专门借款在资本化期间确认的利息收入扣除所支付的利息费用(包括占用<br>其他专门借款而发生的利息支出)之后的差额，全额资本化 |\n\n\
                        问题：{query}'''
            elif doc_label == "SyllabusQA" or "FinTextQA":
                prompt = self.instruction['en']  + '\n' + f'''Please write a section of reference material related to this {doc_label} test question, incorporating the question itself.\n\
                        example:\n\
                        Question: What are the different purchase options for CNOW, and which option would be most beneficial for students planning to take Corporate Tax next semester?\n\
                        Answer: REQUIRED MATERIALS: \n\
                        South-Western Federal Taxation 2023: Individual Income Taxes, 2023 Edition, Hoffman/Young/Raabe/Maloney/Nellen ISBN-13: 978-0-357-71982-4\n\
                        We will also use an on-line module for the weekly assignments, Cengage NOW (CNOW).  CNOW has a new product called CNOW Unlimited.  You can see a video about it here: https://www.cengage.com/student-training/cnowv2/blackboard/ia-no.\n\ 
                        You can purchase access to CNOW and the eBook for $124.99 for the semester from the publisher through the Blackboard link.  This is the only way to purchase the access to CNOW.  You can also rent a text book for $9.99 as well through the Cengage website.  I would suggest doing this since my exams are open book.\n\
                        When you log onto CNOW take a look at the different purchase options.  If you think you will take Corporate Tax next semester you may want to look at the yearlong option for CNOW Unlimited since CNOW is used in Corporate Tax.  If you have any questions let me know before you order.\n\ 
                        Question: {query}'''
                print(prompt)
            if use_langchain is True:
                response = self.langchain_rewriter(retriever=None, query=prompt, model_type = model_type)
            else:
                response = self.rewrite(prompt)
            return response

    def RaFe(self, query, use_langchain, model_type = "gpt-4o",**kwargs):
        prompt = self.instruction['en'] + f"Please rewrite this query, providing a rewritten version, to overcome the limitations of vector-distance-based retrieval systems, thereby enhancing the system's retrieval capabilities. Query: {query}"
        if use_langchain is True:
            response = self.langchain_rewriter(retriever=None, query=prompt, model_type = model_type, **kwargs)
        else:
            response = self.rewrite(prompt)
        return response

    def rewrite(self, prompt):
        response =  hf_inference(prompt, self.tokenizer, self.device, self.model)
        return response
    
    def RR_Rewrite(self, query, doc_label ,use_langchain = False, model_type = "gpt-4o", **kwargs):
        if doc_label == "保荐代理人考试":
            prompt = self.instruction['cn']  + '\n' + "这是一道保荐代理人考试试题，请分析题目的考察意图，以便更有效地检索相关文档。"
            examples = [{"query": "下列关于合营安排的表述正确的是（ ）。\nA合营方增加其持有的一项构成业务的共同经营的利益份额时，如果合营方对该共同经营仍然是共同控制，则合营方之前持有的共同经营的利益份额应按照新增投资日的公允价值计量", 
                        "rewritten_query": "要解答A选项，在保荐代表人的教材中，我最需要查找的是: 1. 关于合营安排的定义和分类；\n2. 合营安排中共同控制的具体判定标准。\n3. 共同经营的会计处理原则"},
                        {"query": "关于会计信息质量要求，下列说法正确的有（ ）。\nA及时性要求企业对于已经发生的交易或事项，应当及时进行确认、计量和报告", 
                        "rewritten_query": "要解答A选项，在保荐代表人的教材中，我最需要查找的是：1. 会计信息质量要求的定义和基本原则；\n2. 会计信息及时性的具体要求;"}]
            few_shot_prompt = ""
            for example in examples:
                few_shot_prompt += f"示例:\n查询: {example['query']}\n重写的查询: {example['rewritten_query']}\n\n"
            few_shot_prompt += f"查询: {query}\n重写的查询:"   
        
        elif doc_label == "SyllabusQA":
            prompt = self.instruction['en']  + '\n' + "The core knowledge points from the document did not appear directly in this question, but rather it was designed to indirectly test the comprehension ability of the respondent. Please analyze the intention behind this question."
            examples = [{"query": "What is the grading scale to get an A in this course?", 
            "rewritten_query":"Academic Assessment (Grading)"},
            {"query": "How is homework assigned in class?", 
            "rewritten_query": "Homework policies"}]
            few_shot_prompt = ""
            for i, example in enumerate(examples):
                few_shot_prompt += f"Example {i}:\nQuestion: {example['query']}\nIntention behind the question: {example['rewritten_query']}\n\n"
            few_shot_prompt += f"My question: {query}\nAnd now find the intention behind the question" 
        elif doc_label == "FinTextQA":
            prompt = self.instruction['en']  + '\n' + "This question does not directly refer to the content of the document, but instead requires the respondent to understand the meaning of the document. Please analyze the intention behind this question."
            examples = [{"query": "What is the purpose of the document?", 
                         "rewritten_query": "The purpose of the document is to provide information about the company's financial statements."},
                         {"query": "What is the balance sheet?", 
                          "rewritten_query": "The balance sheet is a financial statement that shows the company's assets and liabilities."}]
            few_shot_prompt = ""
            for i, example in enumerate(examples):
                few_shot_prompt += f"Example {i}:\nQuestion: {example['query']}\nIntention behind the question: {example['rewritten_query']}\n\n"
            few_shot_prompt += f"My question: {query}\nAnd now find the intention behind the question"
        if use_langchain is False:  
            response = self.rewrite(few_shot_prompt)
        else:
            response = self.langchain_rewriter(retriever=None, query=few_shot_prompt, model_type = model_type, **kwargs)
        return response

    def TOC_rewriter(self, query , use_langchain = False, model_type = "gpt-4o", **kwargs):
        if kwargs["doc_label"] == "保荐代理人考试":
            prompt = self.instruction['cn'] + '\n' + f'''我会提供可能有多种答案的问题，因为它们有不同的可能解释。将给定的问题澄清为几个消除歧义的问题。\n\
                    示例问题：在《哈利·波特》中谁扮演了韦斯莱兄弟？\n\
                    示例消歧：\n\
                    1. 在《哈利·波特》系列书籍中，谁扮演虚构角色弗雷德和乔治·韦斯莱？\n\
                    2. 在《哈利·波特》电影系列中，因扮演弗雷德和乔治·韦斯莱而闻名的英国演员和双胞胎兄弟是谁？\n\
                    3. 在《哈利·波特》系列中，谁是扮演珀西·韦斯莱的演员？\n\
                    问题：{query}'''
        else:
            prompt = self.instruction['en'] + '\n'+ f'''II will provide questions that may have multiple answers because they have different possible interpretations. Clarify the given question into several unambiguous questions.\n\
                    Example question: Who plays the Weasley brothers in Harry Potter?\n\
                    Example disambiguation:\n\
                    1. In the Harry Potter book series, who plays the fictional characters Fred and George Weasley?\n\
                    2. In the Harry Potter movie series, who are the British actors and twin brothers known for playing Fred and George Weasley?\n\
                    3. In the Harry Potter series, who is the actor who plays Percy Weasley?\n\
                    Question: {query}'''
        if use_langchain is False:
            response = self.rewrite(prompt)
        else:
            response = self.langchain_rewriter(retriever=None, query=prompt, model_type = model_type, **kwargs)
        return response

    def RL_rewriter(self):
        '''
        Implemented in Repo: https://github.com/xbmxb/RAG-query-rewriting
        '''
        raise NotImplementedError

    def langchain_rewriter(self,retriever, query, model_type , doc_label = "保荐代理人考试", **kwargs):
        if model_type == 'o1':
            llm = ChatOpenAI(model_name=model_type, 
                                temperature=kwargs["temperature"],
                                top_p=kwargs["top_p"])
        else:
            llm = ChatOpenAI(model_name=model_type, 
                            temperature=0 if "temperature" not in kwargs else kwargs["temperature"])
        output_parser = LineListOutputParser()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"你学识渊博，很擅长解答{doc_label}的考试试题"),
                ("human", "{question}"),
            ]
        )
        LI_chain = ( {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser())      
        response = LI_chain.invoke({"question": query})    
        return response


if __name__ == '__main__':
    llm_rewriter = LLM_Rewriter(model_name="/home/wangqi/Projects/LLaMA-Factory/saves/Qwen2-7B-Chat/lora/train_2024-07-21-18-10-56/checkpoint-200",use_langchain=True)
    question = "I will not able to attend exams from 11october to 4 November, Which exams will i miss?"
    print(llm_rewriter.RaFe(query=question, use_langchain=True, temperature=0.5))
