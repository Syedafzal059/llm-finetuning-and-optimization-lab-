import torch

def generate_text(model, tokenizer, prompt, config):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    #no_grad = switching off the gradient computation
    #we are just infering here no gradient compution required
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config["model"]["max_new_tokens"],
            temperature=config["inference"]["temperature"],
            top_p=config["inference"]["top_p"],
            do_sample=True,
            # do _sample of then model will use bydefault greedy decoding and default temperature
            # here it will use "top_p" and give temperature  
        )
    #Model output is token IDs
    #convert token back to human-readable text
    return tokenizer.decode(output[0], skip_special_tokens=True)

