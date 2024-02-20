# install & Run
```
# install
pip install .
# copy config and edit it.
cp config_template.yaml config.yaml
# run
python main.py

```

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