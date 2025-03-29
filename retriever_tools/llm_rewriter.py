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
                         "rewritten_query": "The purpose of the document is to provide information about the company's financial statements."}]
            few_shot_prompt = ""
            for i, example in enumerate(examples):
                few_shot_prompt += f"Example {i}:\nQuestion: {example['query']}\nIntention behind the question: {example['rewritten_query']}\n\n"
            few_shot_prompt += f"My question: {query}\nAnd now find the intention behind the question"

        elif doc_label == "AmbigQA":
            reasoning_text = "In the context of a play, when we talk about who \"made\" the play, the most common and essential meaning usually points to the person who created the content of the play, that is, the one who \"wrote\" it."\
            "Writing is the fundamental step in creating a play, determining its plot, characters, and dialogue. While \"made\" can have other meanings like producing or directing, in the most straightforward and typical sense related to the origin of a play, "\
            "it corresponds to the act of writing. Therefore, the gold edit of \"Who made the play *The Crucible*?\" is \"Who wrote the play *The Crucible*?\""
            few_shot_prompt = self.instruction['en'] + '\n' + "Query: Who made the play *The Crucible?*" \
                    f"Reasoning:{reasoning_text}\n" \
                    "Rewrite: Who wrote the play *The Crucible?*\n"\
                    "Query: " + query + "\n" + "First, analyze the reasoning behind the query, and then rewrite the query."

        if use_langchain is False:  
            response = self.rewrite(few_shot_prompt)
        else:
            response = self.langchain_rewriter(retriever=None, query=few_shot_prompt, model_type = model_type, **kwargs)
        return response

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
                # ("system", f"你学识渊博，很擅长解答{doc_label}的考试试题"),
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
