import re
import agent.prompts as prompts
from agent.agent_utils import print_with_color, parse_prompt_rsp

def agent_prompt(image_path, short_text, mllm):
    prompt = prompts.task_template_prompt

    while True:
        prompt = re.sub(r"<short_text>", short_text.replace(".",""), prompt)
        status, rsp = mllm.get_model_response(prompt, image_path)
        
        if status:
            sentences_list = parse_prompt_rsp(rsp)
            
            if sentences_list[0] == "ERROR":
                continue
            else:
                for sentence in sentences_list:
                    print_with_color(sentence, 'yellow')
                    
                break
                
    return sentences_list[0], sentences_list[1]

def agent_descriptions(image_path, short_text, mllm):
    prompt = prompts.task_template_description

    while True:
        prompt = re.sub(r"<short_text>", short_text.replace(".",""), prompt)
        status, rsp = mllm.get_model_response(prompt, image_path)
        
        if status:
            sentences_list = parse_prompt_rsp(rsp)
            
            if sentences_list[0] == "ERROR":
                continue
            else:
                for sentence in sentences_list:
                    print_with_color(sentence, 'yellow')
                    
                break
                
    return sentences_list

def agent_object(image_path, short_text, mllm):
    prompt = prompts.task_template_object

    while True:
        prompt = re.sub(r"<short_text>", short_text.replace(".",""), prompt)
        status, rsp = mllm.get_model_response(prompt, image_path)
        
        if status:
            sentences_list = parse_prompt_rsp(rsp)
            
            if sentences_list[0] == "ERROR":
                continue
            else:
                for sentence in sentences_list:
                    print_with_color(sentence, 'yellow')
                    
                break
                
    return sentences_list