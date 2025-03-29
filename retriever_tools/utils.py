import re
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu

def compute_acc(reference, candidate):
    reference_tokens = set(nltk.word_tokenize(reference))
    candidate_tokens = set(nltk.word_tokenize(candidate))
    common_tokens = reference_tokens.intersection(candidate_tokens)
    return len(common_tokens) / len(reference_tokens)

def compute_bleu(reference, candidate):
    reference_tokens = [nltk.word_tokenize(reference)]  
    candidate_tokens = nltk.word_tokenize(candidate)
    return sentence_bleu(reference_tokens, candidate_tokens)

def compute_rouge_l(reference, candidate):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores["rougeL"].fmeasure  



def get_f1_score(precision, recall):
    return 2 * (precision * recall / (precision + recall))

def str_parser(str):
    str = str.replace('\n', '')
    if 'json' in str:
        str = str.replace('json', '')
    str = str.replace('`', '')
    return str


def hf_inference(prompt, tokenizer, device, model):
    messages = [
        # For Gemma Model, system message is not needed
        {"role": "system", "content": f"Your knowledge is profound, and you are very adept at handling professional issues."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding="longest", max_length=4096, truncation=True, 
                                    return_attention_mask=True, add_special_tokens=True)
    model_inputs = model_inputs.to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
        attention_mask=model_inputs['attention_mask'].to(device), do_sample=True, temperature=0.1, 
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def extract_number(s):
    pattern = r'\d+(\.\d+)?'
    
    match = re.search(pattern, s)
    
    if match:
        return match.group()
    else:
        return ""