<div style="display: flex; align-items: center;">  
    <div style="flex-grow: 1;" align="center">  
        <h2 align="center">(Long)LLMLingua: Enhancing Large Language Model Inference via Prompt Compression</h2>  
    </div>  
</div>

<p align="center">
    | <a href="https://llmlingua.com/"><b>Project Page</b></a> | 
    <a href="https://arxiv.org/abs/2310.05736"><b>LLMLingua Paper</b></a> | 
    <a href="https://arxiv.org/abs/2310.06839"><b>LongLLMLingua Paper</b></a> | 
    <a href="https://huggingface.co/spaces/microsoft/LLMLingua"><b>HF Space Demo</b></a> |
</p>

# Install & Run
```
# install
pip install .
# copy config and edit it.
cp config_template.yaml config.yaml
# run
python main.py

```

# API
- path：api/v1/compress_prompt
- method：GET
- params example:
```
{
    "ratio": 0.5,
    "context": [""],
    "question": ""
}
```
- response example:
```
{
    "data": {
        "compressed_prompt": "",
        "origin_tokens": 3701,
        "compressed_tokens": 2778,
        "ratio": "1.3x",
        "saving": ", Saving $0.1 in GPT-4."
    }
}
```

# Example
- [example_zh](example_zh.md)

# Reference
https://llmlingua.com/
https://github.com/microsoft/LLMLingua
https://zhuanlan.zhihu.com/p/660805821