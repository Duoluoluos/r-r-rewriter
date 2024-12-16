from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import json
import os
import pickle
from tqdm import tqdm
from langchain.docstore.document import Document
from typing import List
from langchain_community.retrievers import BM25Retriever
import re
import sys
sys.path.append('.')

class SRTDocumentRetriever:
    def __init__(self, doc_folder, embedding_model = OpenAIEmbeddings(), use_embeddings = True):
        if use_embeddings is False:
            self.vectorstore = None
            self.doc = self.get_doc_from_folder(folder_path=doc_folder)
        else:
            if os.path.exists('database/src_document/serialized_files/doc_cache') is False:
                self.doc = self.get_doc_from_folder(folder_path=doc_folder)
                self.vectorstore = FAISS.from_documents(documents=self.doc, embedding=embedding_model)
                doc_cache = self.vectorstore.serialize_to_bytes()  # serializes the faiss
                with open("database/src_document/serialized_files/doc_cache", 'wb') as p_file:
                    pickle.dump(doc_cache, p_file)
            else:
                with open('database/src_document/serialized_files/doc_cache', 'rb') as p_file:
                    serialized_vs = pickle.load(p_file)
                    self.vectorstore = FAISS.deserialize_from_bytes(embeddings=embedding_model, serialized = serialized_vs, 
                                                                    allow_dangerous_deserialization=True)

    def load_data(self, file_path, mode):
        '''
        Extract document and metadata from src json files
        '''
        if mode == '教材':
            # Load the JSON data from the file
            with open(file_path) as file:
                data = json.load(file)
            # Extract the 'parent' and 'text' values, merge them into 'knowledge', and add to the knowledge_list
            knowledge_list = []
            for item in data.values():
                parent_data = item['parent']
                sub_block = item['sub_block']
                sub_title = sub_block['title']
                if sub_title == '经典例题详解':
                    continue
                for key, value in sub_block.items():
                    if "sub_block" in key or "mid_data" in key:
                        knowledge_list.extend([f"{parent_data} : {sub_title} : {blocks['paragraph']}" for blocks in sub_block[key] if 'paragraph' in blocks.keys()])
            meta_data = {}
            meta_data['source'] = file_path.split('/')[-1]
            if '财务' in file_path:
                meta_data['class'] = '财务'
            elif '法规' in file_path:
                meta_data['class'] = '法规'
            return knowledge_list, meta_data
        
        elif mode == '例题':
            with open(file_path) as file:
                data = json.load(file)
            meta_data = {}
            meta_data['source'] = file_path.split('/')[-1].split('.')[0]
            knowledge_list = []
            knowledge_list.extend([f"章节：{meta_data['source']}\n  题型：{item['instruction']}\n 问题：{item['input']} \n 答案：{item['output']} \n 解析：{item['CoT']}"
                                    for item in data.values() if 'instruction' in item.keys()  and 'input'  
                                    in item.keys() and 'output'  in item.keys() and 'CoT'  in item.keys()]) 
            return knowledge_list, meta_data
        
    def get_doc_from_folder(self, folder_path):
        knowledge_doc = []
        for f_file in tqdm(os.listdir(folder_path)): 
            file_path = os.path.join(folder_path, f_file)
            # load the json file
            knowledge_list, meta_data = self.load_data(file_path, mode = '教材')
            knowledge_doc.extend([Document(page_content=knowledge) for knowledge in knowledge_list])
        return knowledge_doc

    def invoke(self, query, k=3):
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return retriever.invoke(query)


class SyllabusDocRetriever:
    def __init__(self, doc_folder, embedding_model = OpenAIEmbeddings(), use_embeddings = True):
        if use_embeddings is False:
            self.vectorstore = None
            self.doc = self.get_doc_from_folder(folder_path=doc_folder)
        else:
            if os.path.exists('database/SyllabusQA/doc_cache') is False:
                self.doc = self.get_doc_from_folder(folder_path=doc_folder)
                self.vectorstore = FAISS.from_documents(documents=self.doc, embedding=embedding_model)
                doc_cache = self.vectorstore.serialize_to_bytes()  # serializes the faiss
                with open("database/SyllabusQA/doc_cache", 'wb') as p_file:
                    pickle.dump(doc_cache, p_file)
            else:
                with open('database/SyllabusQA/doc_cache', 'rb') as p_file:
                    serialized_vs = pickle.load(p_file)
                    self.vectorstore = FAISS.deserialize_from_bytes(embeddings=embedding_model, serialized = serialized_vs, 
                                                                    allow_dangerous_deserialization=True)
    def get_doc_from_folder(self, folder_path):
        knowledge_doc = []
        for f_file in tqdm(os.listdir(folder_path)): 
            file_path = os.path.join(folder_path, f_file)
            # load the json file
            with open(file_path, 'r', encoding='iso-8859-1') as file:
                content = file.read()
            knowledge_doc.append(Document(page_content = content))
        return knowledge_doc

    def invoke(self, query, k=3):
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return retriever.invoke(query)


class FintextQARetriever:
    def __init__(self, doc_folder, embedding_model = OpenAIEmbeddings(), use_embeddings = True):
        if use_embeddings is False:
            self.vectorstore = None
            self.doc = self.get_doc_from_folder(folder_path=doc_folder)
        else:
            if os.path.exists('database/FintextQA/doc_cache') is False:
                self.doc = self.get_doc_from_folder(folder_path=doc_folder)
                self.vectorstore = FAISS.from_documents(documents=self.doc, embedding=embedding_model)
                doc_cache = self.vectorstore.serialize_to_bytes()  # serializes the faiss
                with open("database/FintextQA/doc_cache", 'wb') as p_file:
                    pickle.dump(doc_cache, p_file)
                    print('fintext vector store created')


class LIRetriever:
    def __init__(self, retriever, k) -> None:
        self.retriever = retriever
        self.k = k
    def invoke(self, query, preprocess = True):
        if preprocess:
            intents = self.extract_intent(query)
        else:
            intents = [query] if not isinstance(query, list) else query
        scope = self.k
        if len(intents) == 0:
            return []
        elif len(intents) < self.k:
            scope = len(intents)
        retrieved_docs = []
        for i in range(scope):
            print(intents[i])
            retrieved_docs.extend(self.retriever.invoke(intents[i], k=self.k//scope))
        return retrieved_docs

    @staticmethod
    def extract_intent(query):
        query = query.replace('\n', ' ')
        match = re.search(r'[：:](.+)', query)
        if not match:
            return []  # 如果没有找到匹配的内容，返回空列表

        intent_part = match.group(1)
        intents = re.split(r'\d+\.\s*|\s*；\s*', intent_part)
        intents = [intent.strip() for intent in intents if intent.strip() != '']
        processed_intents = []
        for intent in intents:
            if ':' in intent:
                processed_intents.append(intent.split(':')[0])
            elif '：' in intent:
                processed_intents.append(intent.split('：')[0])
            elif ' ' in intent:
                processed_intents.append(intent.split(' ')[0])
            else:
                processed_intents.append(intent)
        print(processed_intents)
        return processed_intents
    