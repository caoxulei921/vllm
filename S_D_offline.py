import sys
sys.path.append("/data/framework_vllm/cxl/vllm")
from vllm import LLM, SamplingParams
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompts = [
    "根据临床表现，你需要进一步住院治疗，并抽血化验",
] * 10

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
sampling_params = SamplingParams(top_k=1, max_tokens=50)

llm = LLM(
    model="/data/framework_vllm/models/Qwen2.5-32B-Instruct-AWQ",
    tensor_parallel_size=1,
    enable_chunked_prefill=False,
    speculative_config={
        # "model": "/data/framework_vllm/models/EAGLE-Qwen2.5-32B-Instruct-new0407",
        "model": "/data/framework_vllm/cxl/models/eagle-32b-ori-med/EAGLE-model",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 5,
        "method": "eagle",
    },
    enforce_eager=True
)

#llm = LLM(model='/data/sxh/models/Qwen2.5-32B-Instruct-AWQ')
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
