from transformers import AutoTokenizer
import transformers
import torch

local_model = r'D:/Codes/transformers/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(local_model)
pipeline = transformers.pipeline(
    "text-generation",
    model=local_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

sequences = pipeline(
    'Where was Barack Obama born? \nA.Honolulu\nB.Chicago\nC.Nairobi\nD.I do not know\nAnswer:\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")