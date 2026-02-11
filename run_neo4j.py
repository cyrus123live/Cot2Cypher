from peft import PeftModel, PeftConfig
import modal
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

app = modal.App("llm-thesis-neo4j")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "bitsandbytes",
        "hf-transfer",
    )
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

instruction = (
    "Generate Cypher statement to query a graph database. "
    "Use only the provided relationship types and properties in the schema. \n"
    "Schema: {schema} \n Question: {question}  \n Cypher output: "
)



def prepare_chat_prompt(question, schema) -> list[dict]:
    chat = [
        {
            "role": "user",
            "content": instruction.format(
                schema=schema, question=question
            ),
        }
    ]
    return chat

def _postprocess_output_cypher(output_cypher: str) -> str:
    # Remove any explanation. E.g.  MATCH...\n\n**Explanation:**\n\n -> MATCH...
    # Remove cypher indicator. E.g.```cypher\nMATCH...```` --> MATCH...
    # Note: Possible to have both:
    #   E.g. ```cypher\nMATCH...````\n\n**Explanation:**\n\n --> MATCH...
    partition_by = "**Explanation:**"
    output_cypher, _, _ = output_cypher.partition(partition_by)
    output_cypher = output_cypher.strip("`\n")
    output_cypher = output_cypher.lstrip("cypher\n")
    output_cypher = output_cypher.strip("`\n ")
    return output_cypher

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    gpu=modal.gpu.A10G(),
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=900,
    container_idle_timeout=300,
    keep_warm=1,
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
)
def generate_cypher(question: str, schema: str) -> list[str]:

    # Model
    model_name = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    new_message = prepare_chat_prompt(question=question, schema=schema)
    prompt = tokenizer.apply_chat_template(new_message, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Any other parameters
    model_generate_parameters = {
        "top_p": 0.9,
        "temperature": 0.2,
        "max_new_tokens": 512,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    inputs.to(model.device)
    model.eval()
    with torch.no_grad():
        tokens = model.generate(**inputs, **model_generate_parameters)
        tokens = tokens[:, inputs.input_ids.shape[1] :]
        raw_outputs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        outputs = [_postprocess_output_cypher(output) for output in raw_outputs]
    return outputs

@app.local_entrypoint()
def main():
    question = "What are the movies of Tom Hanks?"
    schema = "(:Actor)-[:ActedIn]->(:Movie)"
    outputs = generate_cypher.remote(question, schema)
    print(outputs)
    # Example:
    # ["MATCH (a:Actor {Name: 'Tom Hanks'})-[:ActedIn]->(m:Movie) RETURN m"]
