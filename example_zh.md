# Compress Prompt
compress model: Qwen1.5-0.5B-Chat-AWQ
response time: 392ms
reference: https://zhuanlan.zhihu.com/p/660805821

compress request:
```
{
    "ratio": 0.2,
    "context": ["我们制作了一个Project Page，来展示现实场景中压缩的cases，包括 RAG、 Online Meeting、CoT 和Code；\n\nLongLLMLingua已经整合到了LlamaIndex Pipeline中，这是一个广泛使用的RAG框架。\n\nTl;DR:\n\nLLMLingua, 利用经过Alignment的well-trained的小的语言模型，例如GPT2-small或者LLaMA-7B，来检测和剔除prompt中的不重要token，将其转化为一种人类很难理解但是LLMs能很好理解的形势。并且这种被压缩的prompt可以直接用在black-box LLMs中，实现最高20倍的压缩，且几乎不影响下游任务性能，尤其是LLMs特有的能力，例如ICL，Reasoning等。\n\nLLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models (EMNLP 2023).\nHuiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang and Lili Qiu\n\nLongLLMLingua, 利用Prompt 压缩增强LLMs在Long Context Scenarios下感知prompt中关键信息的能力，能够有效缓解Lost in the Middle, 十分适合RAG场景中使用。实现每1k个样本节省最高$28.5(GPT-3.5-Turbo, 4的话这个值还能x10)，并且还能提升LLMs的性能。\n\nLongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression (Under Review).\nHuiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang and Lili Qiu\n\n既然是博客，我们就换一种方式从high level角度讲一下我们对这个工作的理解。\n\n首先，感谢 @九号 之前的分享，启发我们在这个topic上面做的一些工作。\n标题以及LLMLingua的名字其实表达的是我们的一个goal，通过某种方式，构造一种专属于LLMs的语言。\n这种语言很有可能人类很难理解，但是LLMs可以很好的理解，从而可以在这种语言上面做一些有趣的事情。\n比如说用更少的比特数传递更多的信息，甚至可以传递给LLMs更高效的信息，从而提升LLMs的性能。\n\n而我们选择了从Prompt压缩这条路来接近这个goal。\n而且通过Prompt压缩来做这件事非常Make Sense，其一是因为Natrual Language本身是冗余的，尤其是口语下，第二是ChatGPT实际上能很好的理解某种被压缩的文字[1].\n如果从LLMs is Compressor 角度来想这个问题，会变得更顺理成章。即prompt中更高置信的部分token完全不需要交给LLMs，他也能从上下文中猜测出来。\n这也就是我们这系列工作的一个核心出发点。\n\n而压缩带来的Inference 加速，API Cost节省都是实现这个goal的一些副产物，只是这些副产物已经非常吸引人了。\n\n而在LongLLMLingua这篇工作中，我们讨论了Long Context Scenoias 下prompt低信息密度的现象，并进一步利用了信息密度及位置与LLMs performance的关系，提出了一种Question-Aware 的方法来增强LLMs 感知关键信息的能力。\n\n接下来我会分别介绍这两个工作。\n\nLLMLingua\n我个人是非常喜欢这篇工作，我们在这篇中做了很多细致的分析，得到了一些很有趣的发现。\n\nPaper里我们试图去回答以下几个问题，不过可能有些有趣的发现因为在较后的位置，甚至是附录，可能会被大家忽略。\n\n我们应该如何去设计一个prompt 压缩算法，从而能够最大化的压缩prompt，同时又不影响LLMs的性能。\n这种被压缩的prompt能直接用在下游任务中吗？\n它的泛化性怎么样？\n有什么证据能证明Black-box LLMs能理解这种被压缩的prompt？\n这种压缩方法的有上界吗？\n为什么不用GPT-X来做这件事？\n第一是如何去系统的设计一个Prompt Compression方法，我们的Goal是能直接用，不需要对LLMs额外的训练，而且能平衡语言完整性和压缩率之间的关系。\n我们想到了OoD里面基于PPL的一系列方法，OoD的信息正好是Prompt给到LLMs的有效信息。\n这条路会有别于Token Pruning/Merge, 亦或是Soft prompt based method, 从而有希望在不对LLMs进行额外训练的情况下，直接将被压缩的prompt用在black-box LLMs上。\n\n但仅仅基于ppl就够了么？我们发现\n\nprompt中不同成分对于压缩的敏感程度是不同的，例如System prompt，question prompt的敏感度更高;\n高压缩率下，过度影响语言完整性，会让LLMs难以理解prompt中的信息；\n部分Context 中的信息是冗余重复的，尤其是ICL场景下;\nPPL 分布会随着压缩的发生而发生改变；\n需要有某种方式让小模型aware Black-box LLMs；\n这也就是我们在LLMLingua中设计了Budget Controller，Iterative Token-level Prompt Compression, Alignment 三个module的原因，具体细节可见paper。\n\n第二个问题也是所有Efficient Method in LLMs都会遇到的问题，不过之前大部分工作也只是在一些传统的Zero-shot QA或者Language Model task上进行测试，为了进一步说明这种被压缩prompt对于下游任务的影响，我们专门从LLMs特有的一些能力出发，评测了ICL，Reasoning，Summarization，和Conversation这些任务。结果显示我们在GSM8K上可以做到20x的压缩比，并且几乎不影响performance。在Summarization和Conversatio的结果也比baseline要优。\n顺带回答第六个问题，其实可以看见Generation-based的方法实际上不能很好的保留精心设计的prompt中的关键信息，它会忽略推理细节，甚至生成一个完全不相关的examples，即使是GPT-4 也很难完成压缩prompt这件事。\n为了证明LLMLingua的泛化性，我们测试了不同small language model 和black-box LLMs，结果显示由于我们的设计GPT2 small size的模型也能取得不错的结果。此外，被压缩的prompt也能在Cluade上取得不错的结果，这也说明了LLMLingua的泛化性。\n我们还做了一个有趣的实验，让GPT-4 去帮助回复被压缩之后的prompt，惊奇的发现，居然可以从那些人类很难理解的文本中几乎完全的恢复出所有细节，如下图完全恢复出了9-steps CoT。不过不同压缩程度的prompt能恢复的细节也不同，这也说明了我们的设计是合理的。\n我们还发现压缩Prompt不仅能节省Input tokens，还能进一步节省20%-30% output tokens.\n我们也尝试使用了更多的压缩率，结果显示即使是利用了LLMLingua，在特别大的压缩率下仍然会出现特别剧烈的performance drop。\n除此之外，我们还将LLMLingua apply到KV-Cache Compression的场景，也能取得不错的performance。\nLongLLMLingua\nLongLLMLingua 出发点和LLMLingua就不太一样了，不只是想要压缩prompt保证少掉精度，而是想要在Long Context Scenarios下，通过提升prompt中的信息密度，从而提升LLMs的性能。\n这个方法特别适合现在普遍使用的Retrieval-Augmented Generation Method。\n\n虽然现在有很多工作让模型能够处理更长的context，但是Context Windows 的增长反而会影响很多下游任务的performance[2];\n其次，之前的工作表面prompt中noise的增多，会影响LLMs的性能;\nLost in the middle 中分析了prompt 中关键信息的位置对于LLMs的性能的影响;\nLong Context Prompt 会带来更多的API Cost和latency;\n综合以上几点，我们觉得Long Context Scenorias 中信息密度是一个非常关键的问题，Prompt Compression可能是其中的一个解决方案。\n但是LLMLingua或者其他Compression-based 的method 并不是一个合适的解决方案，原因是Long Context 中关键信息的密度很低，很有可能prompt本身的信息熵很高，但是不相关，这样压缩prompt反而会引入更多的噪声，从而影响performance。\n\n我们的解决方案是，通过设计了Question-Aware Coarse-fine的方法，让压缩算法能够感知到因为question带来的关键信息分布的变化。\n具体细节可以参考我们的paper。其中Coarse-level的方法甚至还能单独拿来作为一个Retrieval method，取得不错的效果。\n除此之外，我们利用Question-aware 的信息，对Document 进行重排，从而缓解lost in the middle带来的performance影响。可以看到如上右图，在4x压缩率，我们的方法能够略微超过ground truth 位于prompt 开头的结果，从而用更少的API Cost取得更好的结果，缓解lost in the middle 带来的问题。\n我们还设计了dynamic compression ratio 来串联两个粒度方法的信息，设计了一个基于子序列的后处理recovery来恢复被压缩的重要信息。\n\n为了证明我们方法的有效性，我们在Multi-Document QA和两个Long Context Benchmark 中进行了细致的测试。\n其中Multi-Document QA中选用的dataset 更贴合RAG实际场景，k个document均为粗排召回与question十分相关的document。\n可以看到我们的方法通过提高prompt中的信息密度能够有效的提升performance，缓解lost in the middle现象，尤其是经过reorder之后。\n其次，即使单独使用LongLLMLingua中Coarse-level Question-aware的方法，也能取得不错的效果，这也说明了我们的设计有效性。\n我们还测试了Long Context Benchmark中的不同tasks，包括Single-Document, Multi-Document, Summarization, Few-shot Learning, Synthetic和Code补全。\n结果显示我们的方法在Multi-Document QA， Synthetic等任务上提升明显，能够在6x压缩率下获得更好的performance。\n\n除此之外，我们还测试了端到端Latency，和API Cost节省情况。\n结果显示，LongLLMLingua 虽然会比LLMLingua慢，但仍然能拿到实际的端到端加速。\nAPI Cost方面，Long Context Scenorias 下能够节省更多的Cost，最多每1k个样本节省$28.5。\nFQA\n1. 不同LLMs之间Tokenizer 不同会影响压缩效果吗？\n实际上我们用的小模型和大模型之间的Tokenizer完全不同，但是并不影响大模型对compressed prompt的理解。Token-level Compression是为了进一步感知LLMs中对于prompt的token的重要分布。\n2. 你看过那么多压缩的example，你觉得人能总结出某种语法吗？\n我觉得十分困难，Natural Language本身有一定的语法规则，但是LLMLingua所蕴含的是语法+world Knowledge，这种Knowledge是很难由某些特定人完全掌握的。\n3. 为什么LLMs能理解Compressed Prompt？\n我们现在的理解是因为world Knowledge是相同的，不同的LLMs其实都是对于同一个Knowledge Space的逼近，不同LLMs能够逼近的程度不同（可以看成是LLMs的压缩率）。\n4. 传统压缩算法能做Prompt Compression吗？\n实际上我们觉得直接做Prompt Compression比较困难，因为压缩完的prompt很有可能token数量不会减少，而且LLMs并不会很好的理解这种格式的信息。但是可以将传统压缩算法看做是某种Retrieval-based Method 来做Coarse-level的prompt compression。\n5. 能利用不同语言信息熵不同的特点，将prompt转换为信息熵高的语言再压缩吗？\n理论上是可以的，不同语言中的信息熵差异非常大。但是这取决于translation System能够保留住原有信息，且black-box LLMs对于对应language能够有与源语言相似的performance。\n"],
    "question": "1. LLMLingua是什么？\n2. LLMLingua的工作原理是什么？\n3. 介绍一下LLMLingua的各个模块。"
}
```

compress response:
```
{
    "data": {
        "compressed_prompt": "我们制作了一个Project Page，来展示现实场景中压缩的cases，包括 RAG、 Online、CoT 和Code；\n\nLongLLMLingua已经整合到了LlamaIndex Pipeline中，这是一个使用的RAG框架。\n\nTl;DR:\n\nLLua,用经过Alignment的well-trained的小模型，例如GPT2-small或者LLaMA-7B，来检测和除中的不重要token，将其转化为一种人类理解但是LLMs能理解的形势。并且这种被的prompt可以直接用在black-box LLMs，实现最高20的压缩，且几乎不影响下游任务性能，尤其是LLMs特有的能力，例如ICL，Reasoning等。\n\nua: Comingpts foratedference of Large Models (EMNLP023).\nHuiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang and Lili Qiu\n\nua,用Prompt �缩增强LLMs在Long Context Scenarios下感知prompt中关键的能力，能够有效缓解Lost in the Middle, �分适合AG场景中使用。实现每1k个样本节省最高$28.5(GPT-3.5-Tbo, 4的话这个值还能x0)，并且还能提升Ms的性能。\n\nua: Accelerating andancingsenarios via Prompt Compression (Under Review).\n Jiang, Xufang Luo, Dongeng, Lin Yangiu\n\n既然是博客，我们就换一种方式从high level角度讲一下我们对这个理解。\n\n，感谢 @九号 之前的，启发我们在这个topic上面做的一些工作。\n标题以及LLua的名字其实表达的是我们的一个goal，通过某种方式，构造一种专属于LLMs的语言。\n语言很有可能人类但是Ms可以理解，从而可以在语言上面做一些的事情。\n比如说用更的比特数传递，甚至可以传递给Ms更高效的信息，从而提升性能。\n\n而我们选择了从压缩来这个goal。\n而且通过压缩来做这件事非常Make Sense，其一是因为Natrual Language本身是，尤其是口语下，第二是ChatPT实际上能很好的理解某种被压缩的文字[1].\n如果从LLMs is Compressor 角度来想，会变得更顺理章。即prompt中更高置信的部分token完全不需要Ms，他也能从上下文中猜测出来。\n这也就是我们这系列工作的一个核心点。\n\n而压缩带来的Inference 加速，API Cost节省都是实现这个goal的一些副产物，只是产物已经非常了。\n\n而在Longua这篇我们了 Scenoias 下prompt低信息密度的现象，并进一步利用了信息密度及位置与Ms performance的关系，种Question-Aware 的来增强Ms �知关键的能力。\n\n接下来我会分别介绍这两个工作。\n\nLLua\n我个人喜欢这篇，我们篇中很多的分析，得到了一些很有趣的发现。\n\nPaper里我们试图去回答问题，不过可能有些有趣的发现因为在较后的位置，甚至是附录，可能会被大家忽略。\n\n我们应该如何去一个prompt缩算法，从而能够化的压缩prompt，同时又不影响性能。\n这种prompt能在吗？\n它的化性？\n有什么证据能证明Blacks理解prompt？\n这种压缩方法的有上界？\n为什么不用PT-X来做这件事？\n第一是如何去系统的设计一个Prompt Compression方法，我们的是能直接用，不需要对Ms额外的训练，而且能平衡语言性和压缩率关系。\n我们想到了OoD里面基于PPL的一系列，D的信息正好是Prompt给到Ms信息。\n这条路会有别于Tokenuning/Merge, 亦或是Soft prompt based method, 从而有希望在Ms，直接将的prompt用在s上。\n\n但仅仅基于pl就够了？我们发现\n\nprompt中不同成分对于压缩的敏感程度不同的，例如System prompt， prompt的敏感度更高;\n高压缩率，过度影响完整性，会让Ms理解prompt信息；\n部分Context 中的信息是余，尤其是L场景下;\nPL 分布会随着压缩而改变；\n需要有某种方式让模型aware Black-boxs；\n这也就是我们在LL中设计了Budget Controller，Iterative Token-level Prompt Compression, Alignment 三个module的原因，具体细节可见paper。\n\n第二个问题也是所有Efficient Method ins，不过之前大部分工作也只是在一些传统的Zero-shot QA或者Language Model task上进行，为了进一步说明这种压缩prompt对于，我们专门从Ms有能力，了Ling，ization，和Conversation这些任务。结果显示我们在GSM8K上可以做到0压缩比且不影响performance。在ization和Conversatio的结果也比baseline要优。\n顺带回答第六，其实可以看见Generation-based的方法实际上不能很好的保留prompt中的，它会忽略推理，甚至生成一个完全相关的examples，即使是-4 也很难完成压缩prompt。\n为了证明LLua的化我们了不同small language model 和black-box，结果显示由于我们的设计PT2 small size模型也能取得的结果。，prompt也能在Cluade上取得的结果，性。\n我们还了一个，让4��帮助回复压缩，发现，居然可以从那些人类的中几乎完全的恢复出所有，如下图完全恢复出了9-stepsT。不过不同压缩prompt能恢复细节，了设计是合理的。\n还发现压缩Prompt不仅能节省Input还能进一步节省0% tokens.\n我们也尝试使用了更多的压缩，结果显示即使是利用了ua，在特别大的压缩下仍然会出现特别的performance。\n我们还将LLua到KV-Cache Compression的，也能取得不错的performance。\nLongua\nua�发点和ua就一样，不只是想要压缩prompt保证少掉而是想要在，通过提升prompt中的，从而提升。\n这个方法适合现在使用的valmented Generation Method。\n\n虽然现在有很多工作让模型能够处理更的但是Context Windows 的增长反而会影响很多的performance2];\n，之前的工作表面prompt中noise的，会影响性能;\n middle 中分析了prompt 中信息的位置对于的影响;\nLong Context Prompt 会带来更多的 Cost和ency;\n以上我们觉得Long Context Scenorias 中密度是一个非常关键，Prompt Compression可能是的一个。\n但是LLua或者其他Compression-based 的method�不是一个，原因是 Context 中信息的密度，很有可能prompt本身的信息熵，但是不相关，这样压缩prompt会，影响。\n\n我们的是，通过设计了ware Coarse-fine让压缩能够到因为question带来的信息分布的变化。\n具体细节参考我们的paper。其中arse-level的方法甚至还能单独拿来作为一个val，取得。\n我们利用Question-aware 的信息，对Document行重，从而缓解 middle带来的performance。可以看到如上图，在4x压缩率，我们的方法能够略微超过 truth 位于prompt头，从而用的API Cost结果，缓解 middle �。\n还了dynamic compression ratio�串联两个度方法，设计了一个基于子的后处理recovery来恢复压缩。\n\n证明我们方法，我们在-Document QA和两个 Context Benchmark 中进行了。\n其中 QA中的dataset 更合AG，k个document均为粗排召回与question十分相关的。\n可以看到方法通过密度能够有效的提升，缓解 middle，尤其是经过reorder。\n，即使使用Longua中Coarse-level-aware取得，设计。\n还了 Context Benchmark中的不同，包括Document,,-shot, Synthetic和Code全。\n结果显示在， Synthetic上提升，能够在6x下获得更好的。\n\n了端，和 Cost节省。\n，Longua �然会比LLua但能拿到的端端。\n Cost，Longorias 下能够节省更多的，最多。\nFQA\n. 不同LLMs之间Tokenizer同会影响压缩效果？\n实际上我们用模型和之间的Tokenizer，但是影响对。Token-level Compression是为了进一步感知Ms中的token的重要 你看过那么多压缩的，你觉得能出某种？\n我觉得， Language本身但是LLua所蕴含的是语法+world，这种是很难由某些特定人为什么LLMs理解？\n我们现在的理解是因为是，Ms其实都是对于同一个Knowledge Space的逼近，不同LLMs能够逼近的程度不同（可以看成是LLMs的压缩率）。\n4. 传统压缩算法能做Prompt Compression吗？\n实际上我们觉得直接做Prompt Compression比较困难，因为压缩完的prompt很有可能token数量不会减少，而且LLMs并不会很好的理解这种格式的信息。但是可以将传统压缩算法看做是某种Retrieval-based Method 来做Coarse-level的prompt compression。\n5. 能利用不同语言信息熵不同的特点，将prompt转换为信息熵高的语言再压缩吗？\n理论上是可以的，不同语言中的信息熵差异非常大。但是这取决于translation System能够保留住原有信息，且black-box LLMs对于对应language能够有与源语言相似的performance。\n\n\n1. LLMLingua是什么？\n2. LLMLingua的工作原理是什么？\n3. 介绍一下LLMLingua的各个模块。",
        "origin_tokens": 3701,
        "compressed_tokens": 2778,
        "ratio": "1.3x",
        "saving": ", Saving $0.1 in GPT-4."
    }
}
```

# Ask for llm 
question: "1. LLMLingua是什么？\n2. LLMLingua的工作原理是什么？\n3. 介绍一下LLMLingua的各个模块。"
temperature:0

## Qwen1.5-7B-Chat-AWQ

### full_prompt
response_time: 5.47s
```
{
    "id": "cmpl-8058b207cdcb4a9f95c1caff2068a16b",
    "object": "chat.completion",
    "created": 1716483,
    "model": "/mnt/ruby/models/Qwen1.5-7B-Chat-AWQ",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "LLMLingua是一个项目，它利用经过Alignment的well-trained小型语言模型，如GPT2-small或LLaMA-7B，来检测和剔除prompt中的不重要token，然后将其转化为一种人类难以理解但LLMs能很好理解的形式。这种被压缩的prompt可以直接用于black-box LLMs，实现高达20倍的压缩，同时几乎不影响下游任务性能，特别是LLMs的特殊能力，如ICL和Reasoning。LLMLingua的工作原理包括压缩算法、Budget Controller、Iterative Token-level Prompt Compression和Alignment模块。\n\n1. LLMLingua是一个通过压缩prompt来提升大型语言模型性能的工具。\n2. 它通过检测并移除prompt中的冗余信息，同时保持LLMs能理解的关键内容，实现对prompt的高效压缩。\n3. LLMLingua的模块包括：\n   - Budget Controller：控制压缩过程，确保语言完整性与压缩率之间的平衡。\n   - Iterative Token-level Prompt Compression：逐个token进行压缩，根据上下文信息动态调整。\n   - Alignment：利用模型对上下文的理解来识别和保留关键信息。\n   \n这些模块的结合使得LLMLingua能够在长文本场景中有效压缩prompt，同时保持LLMs的性能，并减少API成本。\n"
            },
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 2416,
        "total_tokens": 2688,
        "completion_tokens": 272
    }
}
```

### compressed_prompt
response_time: 4.57s

```
{
    "id": "cmpl-dbe21c8845ab4fb3bdf6776282fd68d1",
    "object": "chat.completion",
    "created": 1716776,
    "model": "/mnt/ruby/models/Qwen1.5-7B-Chat-AWQ",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "1. LLMLingua是论文中提到的一个项目，它使用经过Alignment的well-trained小模型，如GPT2-small或LLaMA-7B，来检测和移除不重要tokens，然后转化为一种人类可理解但LLMs也能理解的形式。这种经过处理的prompt可以直接用于black-box LLMs，实现高压缩比例，同时几乎不影响LLMs的特定能力，如ICL和Reasoning。\n\n2. LLMLingua的工作原理是通过一种压缩算法，它不是简单地减少token数量，而是通过一种策略让LLMs能够从上下文中推断出被移除的部分。这种设计使得prompt在压缩后仍然保持足够的信息，同时减少了输入的比特数，从而提高性能并节省API成本。\n\n3. LLMLingua的模块包括Budget Controller、Iterative Token-level Prompt Compression和Alignment。Budget Controller负责控制压缩的程度，确保关键信息不丢失；Iterative Token-level Prompt Compression是一个迭代过程，逐个token进行压缩决策；Alignment则通过某种方式使模型对黑盒模型保持敏感，理解压缩后的prompt。这些模块共同确保了压缩prompt的高效性和对下游任务的影响最小。\n"
            },
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 1802,
        "total_tokens": 2051,
        "completion_tokens": 249
    }
}
```

## Yi-34B-Chat-AWQ
### full_prompt
response_time: 14.00s
```
{
    "id": "chatcmpl-38aST8EMrhAyVfXnp2XhPY",
    "object": "chat.completion",
    "created": 1708449863,
    "model": "Yi-34B-Chat-AWQ",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "1. LLMLingua是什么？\nLLMLingua是一种用于压缩大型语言模型（LLMs）prompt的技术。它旨在通过去除prompt中的不重要token，将prompt转换为一种人类难以理解但LLMs可以高效理解的格式。这种压缩后的prompt可以直接用于black-box LLMs，实现高压缩比的同时，保持甚至提升下游任务的性能。\n\n2. LLMLingua的工作原理是什么？\nLLMLingua的工作原理是基于对prompt中token的重要性和信息密度的分析。它使用了一种称为“预算控制器”（Budget Controller）的算法来确定哪些token可以被安全地移除，同时保持prompt的完整性。通过迭代式的token-level prompt compression和alignment技术，LLMLingua能够将prompt压缩到极致，同时保持LLMs对其的理解能力。\n\n3. 介绍一下LLMLingua的各个模块。\nLLMLingua主要包括以下几个模块：\n\n   - **预算控制器（Budget Controller）**：这个模块负责评估prompt中每个token的重要性，并决定哪些token可以被移除以实现最大程度的压缩。\n   - **迭代式token-level prompt compression**：这个模块通过反复迭代，逐步移除不重要的token，同时保持prompt的完整性。\n   - **对齐（Alignment）**：这个模块确保压缩后的prompt与原始prompt在语义上保持一致，从而保证LLMs能够正确地理解和执行压缩后的prompt。\n\n通过这些模块的协同工作，LLMLingua能够实现高达20倍的prompt压缩比，同时保持LLMs在各种下游任务中的性能，尤其是那些依赖于LLMs特有能力，如ICL（in-context learning）和rea"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 2574,
        "total_tokens": 2945,
        "completion_tokens": 371
    }
}
```

### compressed_prompt
response_time: 11.38s
```
{
    "id": "chatcmpl-MbedzesGMK4Zx5K9TEwg4p",
    "object": "chat.completion",
    "created": 1708449789,
    "model": "Yi-34B-Chat-AWQ",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "1. LLMLingua是一个用于理解和增强大型语言模型（LLMs）在长上下文场景下性能的研究项目。该项目旨在通过压缩prompt（即模型的输入）来提高模型的效率和性能，同时保持或增强模型的关键能力，如推理、理解和生成。\n\n2. LLMLingua的工作原理是基于对prompt中信息密度的分析。项目团队发现，并非prompt中的所有token对于模型的理解和生成都是同等重要的。因此，他们提出了一种迭代式的token-level prompt compression方法，通过去除不重要的token来减少prompt的长度，同时保持模型的关键能力。这种方法不依赖于对LLMs的额外训练，而是直接在prompt的设计层面进行优化。\n\n3. LLMLingua的各个模块包括：\n   - Budget Controller：这个模块负责在压缩过程中平衡语言性和压缩率的关系，确保prompt在压缩后仍然保持足够的语言信息。\n   - Iterative Token-level Prompt Compression：这是一个核心模块，它通过迭代的方式去除prompt中不重要的token，同时保持模型的理解和生成能力。\n   - Alignment：这个模块用于确保压缩后的prompt与原始prompt在语义上保持一致，从而不影响模型的性能。\n\n通过这些模块的协同工作，LLMLingua能够实现最高20倍的prompt压缩，同时保持或提高模型的性能，尤其是在那些需要长"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 1985,
        "total_tokens": 2284,
        "completion_tokens": 299
    }
}
```
