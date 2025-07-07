# 提交历史

[Eagle github commit](https://github.com/caoxulei921/vllm/commit/c30e2dac9073c7f8422d422722bdf1c31776c103#diff-868acbec2d5d6b566c1f5c29e9a17430615dac9fc40d68d9544ba8b628cb4d1f)

# 开发逻辑
利用vLLM原有给speculative decoding中bounds token的逻辑，通过expand execute_model_req来达到多个tokens在一轮自回归中小模型也具备并行生成next token能力。

原先k个accept的token在下一轮draft中会被expand为[last_accepted, bounds]，如果last_accepted的token是特殊词，且长度为n，则会被expand为[last_accepted_1, last_accepted_2,..,last_accepted_n,bounds]。

当遇到特殊词被触发后终止draft（这个逻辑可以调整），因此eagle模型一轮draft生成的长度非固定位动态。

# 开启Debug
通过修改部分日志，可观察完整的生产过程。

## 统计接受率
打开以下文件的注释部分， vllm/model_executor/layers/rejection_sampler.py
```python
    ...
    logger.info(f'accepted as {accepted}')
    ...
    logger.info (f"** ori accepted: {accepted} with selected_draft_probs: {selected_draft_probs}")
    ....
    logger.info (f"** modify accepted: {accepted}, pos_idx: {pos_idx}")
```

## 观察生成过程，验证逻辑
修改以下文件开关， vllm/spec_decode/draft_model_runner.py
```python
    debug_advance_input = True
```
# 注意事项

## 转移词表
新增转移词表 
```python
llm = LLM(
    ...
    speculative_config={
        "vocab_trans_dict": "/data/framework_vllm/zch/dict.json" 
    },
)
```

## 新增大小词表配置 
```
    "TRUE_EXPAND_VOCAB_SIZE": 4555,
    "TRUE_ORIGIN_VOCAB_SIZE": 151665,
    "PAD_ORIGIN_VOCAB_SIZE": 152064
```

## Sampling配置
注意temperature和top_p配置
```
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=xxx
)
```

## KV Block预分配 配置修改
speculative decoding会根据num_lookahead_slots在一轮自回归中提前分配好block，详见`ensure_num_empty_slots`函数。
因为基于大小词表方案一轮draft生成的tokens不固定，取决于生成的词表，因此需新增参数`speculative_append_slots_len`,设置为最大特殊词长度-1。

```python
llm = LLM(
    ...
    speculative_append_slots_len=4,  ##特殊词词表最大长度-1
    speculative_config={
        ...
    },
)
```

<details>
  <summary> function ensure_num_empty_slots</summary>
/vllm/vllm/core/block/block_table.py
```python
    def ensure_num_empty_slots(self,
                               num_empty_slots: int,
                               extra_hash: Optional[int] = None) -> None:
        """Ensures that the BlockTable has at least the specified number of
        empty slots available.

        This method checks if the BlockTable has enough empty slots (i.e.,
        available space) to accommodate the requested number of tokens. If not,
        it allocates additional blocks on the GPU to ensure that the required
        number of empty slots is available.

        Args:
            num_empty_slots (int): The minimum number of empty slots required.
            extra_hash (Optional[int]): The hash value of additional
                factors such as adapters that influence the block, apart
                from the token_ids.
        """
        # Currently the block table only supports
        # appending tokens to GPU blocks.
        device = Device.GPU
        assert self._is_allocated

        if self._num_empty_slots >= num_empty_slots:
            return

        slots_to_allocate = num_empty_slots - self._num_empty_slots
        blocks_to_allocate = cdiv(slots_to_allocate, self._block_size)

        for _ in range(blocks_to_allocate):
            assert len(self._blocks) > 0
            self._blocks.append(
                self._allocator.allocate_mutable_block(
                    prev_block=self._blocks[-1],
                    device=device,
                    extra_hash=extra_hash))

</details>```




# 附录
## offline 脚本
```python
import sys
sys.path.append("/data/framework_vllm/cxl/vllm")
from vllm import LLM, SamplingParams
import os



prompts = ["system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n请编辑一篇关于紫珠叶的文献综述。<|im_end|>\n<|im_start|>assistant\n"]

system_prompt = "你是一个医疗领域的人工智能助手。问："



sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=512
)


llm = LLM(
    model="/data/framework_vllm/models/Qwen2.5-32B-Instruct-AWQ",
    # model = '/data/framework_vllm/models/Qwen2-7B-Instruct/',
    tensor_parallel_size=1,
    enable_chunked_prefill=False,
    disable_log_stats=False,
    speculative_append_slots_len=4,  ##特殊词词表最大长度-1
    speculative_config={
        "model" :"/data/framework_vllm/cxl/checkpoint/vllm-checkpoint-new_mapping-0630/",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 3,
        "method": "eagle",
        "vocab_trans_dict": "/data/framework_vllm/zch/dict.json" 
    },
    enforce_eager=True,
    gpu_memory_utilization=0.5
)


outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


## torch转vllm模型
```python
from safetensors.torch import load_file, save_file
import os
import torch
# 加载模型参数 torch
model_path = "/data/framework_vllm/sxh/EAGLE/sxh/checkpoint-new_mapping-0630/state_17/"
# 保存模型参数 vllm
save_model_path = "/data/framework_vllm/cxl/checkpoint/vllm-checkpoint-new_mapping-0630/"

##model.safetensors
state_dict = load_file(os.path.join(model_path, "model.safetensors"))
# 遍历参数并打印
print("**Ori model.safetensors dict as:")
for name, tensor in state_dict.items():
    print(f"{name}: shape = {tuple(tensor.shape)}")

print("**Begin convert model.safetensors")
frozen_embed = state_dict["frozen_embed.weight"]
trainable_embed = state_dict["trainable_embed.weight"]

embed_tokens_weight = torch.cat(
    [frozen_embed, trainable_embed],
    dim=0
)
del state_dict["frozen_embed.weight"]
del state_dict["trainable_embed.weight"]
state_dict["embed_tokens.weight"] = embed_tokens_weight
print("**End convert model.safetensors")
print("\n**New model.safetensors dict as:")
for name, tensor in state_dict.items():
    print(f"{name}: shape = {tuple(tensor.shape)}")

save_file(state_dict, (os.path.join(save_model_path, "model.safetensors")))


##model.safetensors
lm_state_dict = load_file(os.path.join(model_path, "model_1.safetensors"))
lm_state_dict2 = load_file(os.path.join(model_path, "model_2.safetensors"))
# 遍历参数并打印
print("**Ori model_1.safetensors dict as:")
for name, tensor in lm_state_dict.items():
    print(f"{name}: shape = {tuple(tensor.shape)}")

lm = lm_state_dict["weight"]
print("**Ori model_2.safetensors dict as:")
for name, tensor in lm_state_dict2.items():
    print(f"{name}: shape = {tuple(tensor.shape)}")
lm2 = lm_state_dict2["weight"]
lm_head = torch.cat(
    [lm, lm2],
    dim=0
)


save_file({"lm_head": lm_head}, (os.path.join(save_model_path, "lm_head.safetensors")))
```