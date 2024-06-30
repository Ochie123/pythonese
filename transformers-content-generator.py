from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from transformers import logging as t_logging

#logging.basicConfig(level=logging.DEBUG)  # Standard Python logging
#t_logging.set_verbosity_debug()           # Transformers logging

tokenizer = AutoTokenizer.from_pretrained("aisquared/dlite-v1-355m")
model = AutoModelForCausalLM.from_pretrained("aisquared/dlite-v1-355m")

generate_text = pipeline("text-generation", model=model, tokenizer=tokenizer,
                         torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

result = generate_text("What is Facebook's transformer model.",
                       #max_length=50,  # Aim for approximately 50 words
                       num_return_sequences=3,
                       pad_token_id=tokenizer.eos_token_id,
                       #truncation=True
                      )
print(result)