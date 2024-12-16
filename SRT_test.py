from retriever_tools.doc_retriever import SRTDocumentRetriever, LIRetriever
from retriever_tools.llm_rewriter import LLM_Rewriter
from datetime import datetime
import os
import json
import tqdm
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever_tools.utils import hf_inference

class LangchainHandler:
    def __init__(self, llm_type, exp_db, textbook_db, **kwargs):
        self.llm = self.create_llm(llm_type, **kwargs)
        if exp_db:
            self.exp_retriever = exp_db.as_retriever()
        if textbook_db:
            self.tb_retriever = textbook_db.as_retriever()
        self.read_chain = self.create_read_chain()
        self.aggregation_chain = self.create_aggregation_chain()

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
        if type == 'reader':
            return ChatPromptTemplate.from_messages([
                ("system", "你是一位经济学家，请你判断这道题目的这个选项是否正确，我会提供参考资料"),
                ("human", "相应的参考资料如下：{context}。问题如下：{question}。"),
            ])
        elif type == 'aggregator':
            return ChatPromptTemplate.from_messages([
                ("system", "你是一位经济学家，请总结你之前的解析，给我提供最终答案"),
                ("human", "问题如下：{question}。\n你之前的解析如下:{analysis}\n\n \
                 请将你的最终答案用json字典格式表示，字典的键为'答案'，值为你选择的选项。例如：{{\"答案\":\"A,C,D\"}}"),
            ])      
        


    def create_read_chain(self):
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.build_prompt(type='reader')
            | self.llm
            | StrOutputParser()
        )
    
    def create_aggregation_chain(self):
        return (
            {"question": RunnablePassthrough(), "analysis": RunnablePassthrough()}
            | self.build_prompt(type='aggregator')
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)





class hfhandler:
    def __init__(self, model_name, device = 'cuda', use_langchain = False) -> None:
        self.model_path = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def invoke(self, func, **kwargs):
        if func == "reader":
            context_value = kwargs["context"]
            question_value = kwargs["question"]
            prompt = f"相应的参考资料如下：{context_value}。问题如下：{question_value}。"
        elif func == "aggregator":
            question_value = kwargs["question"]
            analysis_value = kwargs["analysis"]
            prompt = f"问题如下：{question_value}。\n你之前的解析如下:{analysis_value}\n\n请将你的最终答案用json字典格式表示，字典的键为'答案'，值为你选择的选项。例如：{{\"答案\":\"A,C,D\"}}"
        response =  hf_inference(prompt, self.tokenizer, self.device, self.model)
        return response
    



class ExamValidator:
    def __init__(self, dpr_chain, read_chain, agg_chain, prlm_path, rewrite = True):
        self.read_chain = read_chain
        self.agg_chain = agg_chain
        self.current_date = datetime.now().strftime("%m%d")
        if rewrite:
            self.rewriter = LLM_Rewriter(prlm_path, use_langchain=config["use_langchain"])
        self.dpr_chain = dpr_chain

    @staticmethod
    def calculate_score(str1, str2):
        list_1 = [char for char in str1 if char.isalpha()]
        list_2 = [char for char in str2 if char.isalpha()]
        set_1 = set(list_1)
        set_2 = set(list_2)
        if set_1 == set_2:
            return 1  
        elif set_1.issubset(set_2):
            return 1  
        else:
            return 0  

    @staticmethod
    def split_question_and_options(input_str):
        first_option_index = input_str.find("A")
        # 如果找不到任何选项，则返回原始字符串作为题干，空列表作为选项
        if first_option_index == -1:
            return input_str, []
        question_stem = input_str[:first_option_index - 1].strip()  # 减去1是为了去掉"\n"或空格，并去除两边空白
        options = []
        for letter in ['A', 'B', 'C', 'D', 'E', 'F']:  
            option_start = input_str.find(f"{letter}")
            if option_start == -1:
                break
            # 查找下一个选项的起始位置或字符串结束位置
            next_option_start = input_str.find(f"{chr(ord(letter) + 1)}", option_start)
            if next_option_start == -1:
                next_option_start = len(input_str)  # 如果没有下一个选项，则使用字符串的末尾
            option = input_str[option_start:next_option_start].strip()
            options.append(option)
        
        return question_stem, options

    def validate(self, json_file, k=3):
        log_dic = dict()
        log_dic_w = dict()
        q_dir = os.path.join(exam_folder, json_file)
        with open(q_dir, "r") as f:
            data = json.load(f)
        input_q = [item['input'] for item in data]
        output_gt = [item['output'] for item in data]
        final_score = 0
        num = 0
        for i in tqdm.tqdm(range(len(input_q))):
            num += 1
            question = input_q[i]
            question_stem, options = self.split_question_and_options(question)
            # print(question_stem)
            # print(options)
            analysis = ""
            for j, option in enumerate(options):
                new_key = f"问题{i+1}_选项{chr(ord('A') + j)}"
                # Query Decomposiiton (Set Rewrite = True)
                rewrited_query = self.rewriter.RR_Rewrite(question_stem + '\n' + option, use_langchain=config["use_langchain"], doc_label = "保荐代理人考试")
                log_dic_w[new_key] = rewrited_query
                retrieved_doc = ""
                docs = self.dpr_chain.invoke(rewrited_query, k = k)
                for t in range(k):               
                    retrieved_doc += docs[t].page_content + "\n\n"
                log_dic[new_key] = retrieved_doc
                analysis += self.read_chain.invoke({"context": retrieved_doc, "question": question})

                #############################################################
                # End-to-end Solving (Set Rewrite = False)
                #analysis += self.read_chain.invoke(func = "reader", question = question_stem + '\n' + option, context = "")

                #############################################################

                # Pseduo Doc Generation (Set Rewrite = True)
                # rewrited_query = self.rewriter.HyDE(question_stem + '\n' + option)
                # log_dic[new_key] = rewrited_query
                # retrieved_doc = ""
                # docs = self.dpr_chain.invoke(rewrited_query)
                # for t in range(k):               
                #     retrieved_doc += docs[t].page_content + "\n\n"
                # log_dic[new_key] = retrieved_doc
                # analysis += self.read_chain.invoke({"context": retrieved_doc, "question": question})

                #############################################################
            # langchain aggregator
            res = self.agg_chain.invoke({"question": question, "analysis": analysis})

            #############################################################

            # hf aggregator
            # res = self.agg_chain.invoke(func = "aggregator", question = question, analysis = analysis)
            try:
                res_str = json.loads(res)["答案"]
            except:
                res_str = res
            score = self.calculate_score(res_str, output_gt[i])
            log_dic[f"问题{i+1}作答情况"] = f"问题:{question}\n是否正确:{score}"
            if score > 0 :
                final_score += score

        if num > 0:
            print(f"{json_file}: 正确率:{final_score/num}\n")
            log_dic[f"作答情况"] = f"正确率:{final_score/num}"
        with open(f"log/SRT_Test.json", "w",encoding='utf-8') as f:
            json.dump(log_dic, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    langchain_rewriter_llm = config["langchain_rewriter_type"]
    langchain_reader_llm = config["langchain_reader_type"]
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    # Configuration
    os.environ["TOKENIZERS_PARALLELISM"]="True"
    json_file = 'SRC_test_2024.json'
    doc_path = 'database/src_document/json_files/doc'
    PrLM_path = config["pretrained_rewriter_model"]
    if config["use_langchain"] is False:
        print(f"Using Pretrained Rewriter model: {PrLM_path}")
    exam_folder = 'database/src_document/json_files/test/exam'
    # Initialization
    base_retriever = SRTDocumentRetriever(doc_path, use_embeddings=True)
    textbook_db = base_retriever.vectorstore
    exp_db = None
    # Create model handler and RAG chain
    model_handler = LangchainHandler(langchain_reader_llm, exp_db, textbook_db, **config["model_kwargs"])
    # Validate exams
    validator = ExamValidator(dpr_chain = base_retriever, read_chain = model_handler.read_chain, agg_chain = model_handler.aggregation_chain, prlm_path = PrLM_path, rewrite=True)
    validator.validate(json_file)
