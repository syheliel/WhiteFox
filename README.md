# ![Project logo](assets/logo.svg) WhiteFox: White-box Compiler Fuzzing Empowered by Large Language Models

<p align="left">
    <a href="https://arxiv.org/abs/2310.15991"><img src="https://img.shields.io/badge/arXiv-2310.15991-b31b1b.svg?style=for-the-badge">
</p>

> [!IMPORTANT]
> We are keeping improving the documents and adding more implementation details. Please stay tuned at [README-DEV.md](README-DEV.md) for more information.

![Framework](assets/framework.svg)

## About

* 🦊**WhiteFox** is the first white-box compiler fuzzer using LLMs with source-code information to test compiler optimization.
* WhiteFox adopts a dual-model framework: (i) an analysis LLM examines the low-level optimization source code and produces requirements on the high-level test programs that can trigger the optimization; (ii) a generation LLM produces test programs based on the summarized requirements.

> [!IMPORTANT]
> * **WhiteFox** has detected **96** bugs 🐛 in the PyTorch Inductor, Tensorflow XLA, TensorFlow Lite and LLVM, with 80 confirmed as previously unknown and **61** of them are fixed.

We provide a list of confirmed bug reports in [bugs](bugs.csv).

## 🔨 Getting Started

### Prerequisites

1. Python version >= 3.10.0 (It must support f-string.)
    - highly recommend to use Python 3.9
2. Check our dependent python libraries in requirements.txt and install with:
    - pip install -r requirements.txt
3. Install StarCoder
    - Please follow the instructions in [StarCoder](https://huggingface.co/bigcode/starcoder).

### Running WhiteFox

#### Step 1: Request Summarization

The prompts for NL generation are in [Prompts](Prompts) with the format `Prompts/{compiler}/src2req/{name}.txt`.

If you want to generate the prompt by you own, take the prompt for `torch-inductor` as an example:

```bash
bash scripts/whitefox-torch-prompt-gen-src2req.sh ./prompt-1/
```
the generated prompts will be in `prompts-generated` by default.

after getting the prompts, you can run the following command to generate the requirements:

```bash
# python gpt4.py --prompt-dir=prompts/torch-inductor/src2req \ 
#    --outdir=./requirements-2/torch-inductor/ \
#    --temperature=0.0 \
#    --batch-size=1
```

```bash
python gpt4.py --prompt-dir=prompt-1/torch-inductor/req2nl \
    --outdir=./requirement-2/torch-inductor/ \
    --temperature=0.0 \
    --batch-size=10
```
before running the command, please put your openai api key in `openai.key`:

```bash
echo {api_key} > openai.key
```

#### step 2: test generation
first, you need to generate the prompts for the test generation based on the requirements:

```bash
bash scripts/whitefox-torch-prompt-gen-req2test.sh ./requirement-2/torch-inductor ./prompt-3
```
the generated prompts will be in `prompts-generated` by default.


or you can use the prompts we generated in [prompts](prompts) with the format `prompts/{compiler}/req2test/{name}.txt`.

we leverage [starcoder](https://huggingface.co/bigcode/starcoder) to generate the tests based on the prompts.


##### [option 1]: local mode (recommended!)

we recoomend to use the local mode to generate the tests, which utilizes [vllm](https://github.com/vllm-project/vllm).

you can execute the following command to generate the tests:

```bash
# python starcoder_gen.py --hf-home={path-to-huggingface} --hf-cache={path-to-huggingface-cache} --prompt-dir=prompts/torch-inductor/req2test ----output-dir=starcoder-generated --num=10
```

```bash
python gpt4_gen.py --prompt-dir=prompt-3/torch-inductor --output-dir=gencode-4 --num=10
```

the generated tests will be in `starcoder-generated`.

##### [option 2]: server mode

you can execute the following command to generate the tests:

1. run the starcoder server:

```bash
python starcoder_service.py --hf-home={path-to-huggingface} --hf-cache={path-to-huggingface-cache} --prompt-dir=starcoder-prompts --outdir=starcoder-generated --device='cuda:0' --num=10 --batch_size=10
```

2. put the prompts in `starcoder-prompts` and the generated tests will be in `starcoder-generated`.

```bash
mkdir starcoder-prompts/torch-inductor
cp -r prompts/torch-inductor/req2test starcoder-prompts/torch-inductor/
```

#### step 3: test execution

you can execute the following command to execute the tests:

```bash
python -m torch-exec.run_torch --input-dir=./gencode-4/ --res-dir=result-4
```

The output of the execution will be in `torch-exec/_results-torch`.

# TLDR
```bash
bash scripts/whitefox-torch-prompt-gen-src2req.sh ./prompt-1/
python gpt4.py --prompt-dir=prompt-1/torch-inductor \
    --outdir=./requirement-2/torch-inductor/ \
    --temperature=0.0 \
    --batch-size=10
bash scripts/whitefox-torch-prompt-gen-req2test.sh ./requirement-2/torch-inductor ./prompt-3
python gpt4_gen.py --prompt-dir=prompt-3/torch-inductor --output-dir=gencode-4 --num=10
python -m torch-exec.run_torch --input-dir=./gencode-4/ --res-dir=result-4
```