# Moonshot
![moonshot_graphic.png](img%2Fmoonshot_graphic.png)

## Automatic (Zero-code, near Zero-config) Hyperpersonalization

Alpha-stage library that could almost replace your customer support department.
## Features & Roadmap:
- [x] Intelligent information retrieval and summarization for personalized recommendations based on user context
- [x] Automatic detection of relevant user tables
- [x] Automatic table filtering & summarization
- [x] Automatic collection ANN search & summarization
- [x] Web crawler with NLP-based content summarization, keyword extraction, and tagging
- [x] Support for native C* driver and JSON API (Astrapy)
- [x] LangSmith integrated
- [x] Cloud native
- [x] Async
- [ ] ColBERTv2 support (coming soon)
- [ ] Graph neural network enhanced recommendations
- [ ] LangServe compatible
- [ ] Autonomous agent actions
- [ ] Kubernetes native
- [ ] REST API support
- [ ] Multi-modal interactivity
- [ ] Flow visualization and admin control
- [ ] TLS, JWT Token-based auth, and RBAC w/ delegated management
- [ ] Intelligent document discovery and automatic mapping/ingestion
- [ ] Automatic query and LLM error handling & self-correction
- [ ] Real-time data discovery
- [ ] Graph-based prompt construction
- [ ] Automatic data ingestion from S3, GCP Cloud Storage, and Azure Blob Storage
- [ ] VertexAI support

## Hyperpersonalization architecture:
![moonshot_architecture_qa](img/moonshot_architecture_qa.png)

## Contributing
Start by walking through the code in this method: [Personalization flow](https://github.com/devinbost/moonshot/blob/a62c38e8c68e88d78545797560cf3e760d17f87e/Chatbot.py#L178) 

## Feedback welcome!
We'd love to get feedback on what we can do to improve this library. 
Feedback should be created as Issues on this github repo.

Don't forget to **star this repo**!
### Usage:
Start from UserInterface.py

Most easily navigated in PyCharm

Reach out to Devin for more info if interested.

## Setup

### Steps to get running:

#### Clone repo and cd into it:
```commandline
git clone git@github.com:devinbost/moonshot.git
cd moonshot
```

#### Install Conda:
You can install via graphical installer for Miniconda here: https://docs.anaconda.com/free/miniconda/miniconda-install/

#### Setup conda environment
```commandline
conda create -n moonshot_dev python=3.11.7 pip 
```
```commandline
conda activate moonshot_dev
pip install -r requirements.txt
```

### Setup Astra DB
Follow the steps to setup your Astra DB and table. There are some guided instructions you can follow if you scroll down to the "Setup Astra" section of this Colab notebook:
https://colab.research.google.com/drive/1ABgDi6h0mHfkbTizTedRGioUMmiBh3RL#scrollTo=0zzbA6QTHxvo

A CQL script has been provided (scripts/setup_tables.cql) with some starter data that you can run in AstraDB
To run this script, open your Astra portal, navigate to your database, and click the CQL Console button:
![cql_console.png](img/cql_console.png)
In the example script, we assume your namespace is named "telecom". To create it, navigate to the Create Namespace box:
![create_namespace.png](img/create_namespace.png).

## Configuring PyCharm:
Add interpreter:
![interpreter.png](img/interpreter.png)

Select the new conda environment and click OK:
![setup_interpreter.png](img/setup_interpreter.png)

Create a new runtime configuration for running Streamlit:
![create_runtime_config.png](img/create_runtime_config.png)

Click + sign to add new Python configuration:
![add_python_runtime_config.png](img/add_python_runtime_config.png)

Set your env and bin path to point to the new environment.
Also, be sure to set `run core/main.py` and set your working directory to where your repo is running from.
![set_runtime_config_path.png](img/set_runtime_config_path.png)

Set your environment variables.
Some of these environment variables are used by `core/VectorStoreFactory.py` to setup the drivers for the different experiences we will use in this workshop.
Other environment variables are loaded by `core/ConfigLoader.py` and used by `core/LLMFactory.py` to setup your LLM(s).
At a minimum, you will need:
- `ASTRA_TOKEN`
- `ASTRA_ENDPOINT`
- `SECURE_BUNDLE_PATH` (explained below)

The Astra token and endpoint can be obtained from your Astra portal:
![Astra_token_and_endpoint.png](img/Astra_token_and_endpoint.png)

To make use of the text2cql feature of this library (not optional), you will also need to specify:
- `SECURE_BUNDLE_PATH`
which should be the absolute path to a directory containing the secure bundle, which you can download here:
![secure_bundle.png](img/secure_bundle.png)

We recommend creating a directory named "scratch" in this repo and storing the secure bundle there.
This directory is currently specified in .gitignore as an ignored directory to prevent repo commits from accidentally including the secure bundle.

If using OpenAI, you will also need to define:
- `OPENAI_API_KEY`
If using IBM watsonx, you will need to define:
- `IBM_API_SECRET`
- `IBM_PROJECT_ID`

These can be obtained by following the instructions provided by those platforms.

Optionally, if you want integration with LangSmith, you will want to add these variables:
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_ENDPOINT`
- `LANGCHAIN_PROJECT`
- `LANGCHAIN_TRACING_V2` (set to "true")
These LangSmith variable values can be obtained as per LangChain documentation. 

Also, make sure that the content and source roots are added to PYTHONPATH so you don't run into import problems.
You can set those here:
![python_path.png](img/python_path.png)

Finally, apply the changes and click OK.


## Upcoming API changes
- Refactoring builder/factory classes to actually use builder or factory method patterns
- Build abstraction over prompt creation to reduce duplication
- Remove old/unused code
- Refactor dependency injection
- Rewrite graph to build LCEL chains/prompts and deprecate reflection API
- Rewrite tests to use proper mocks,stubs, and fakes 
- Parallelize chains to improve performance
- Consolidate configs


## Research influences:
From the work of Dong, X. et al. (2023), the most common source of errors was incorrect selection of table or column names. 
For this reason, we used a semi-structured approach to code generation where we used a hybrid approach that combined:
1. a specific prompt/chain for determining relevant tables 
2. a semi-structured approach where some of the code body was already provided
This hybrid approach is easier to control for security and auth purposes as well since authorization can be checked before code is executed.

Dong, X., Zhang, C., Ge, Y., Mao, Y., Gao, Y., Lin, J., & Lou, D. (2023). C3: Zero-shot Text-to-SQL with ChatGPT. arXiv preprint arXiv:2307.07306.
