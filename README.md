# Moonshot
![moonshot_graphic.png](img%2Fmoonshot_graphic.png)

## Automatic (Zero-code) Hyperpersonalization

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

## Hyperpersonalization architecture:
![Example moonshot pipeline.png](img%2FExample%20moonshot%20pipeline.png)

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
Setup your PyCharm to run Streamlit like this:
![img.png](img%2Fimg.png)

The following parameters are currently configurable as environment variables:
```commandline
ASTRA_DB_TOKEN_BASED_PASSWORD=AstraCS:...;
IBM_API_SECRET=...; (optional)
IBM_PROJECT_ID=...; (optional)
KEYSPACE=openai;
OPENAI_API_KEY=...;
PROVIDER=OPENAI;
TABLE_NAME=mytable; 
DATABASE_NAME=my_db;
LANGCHAIN_API_KEY=...; (for LangSmith)
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com; (for LangSmith)
LANGCHAIN_PROJECT=my-proj-name; (for LangSmith)
LANGCHAIN_TRACING_V2=true; (for LangSmith)
PYTHONUNBUFFERED=1; (recommended in general)
ASTRA_DB_API_ENDPOINT=https://my-id-my-region.apps.astra.datastax.com (copy from Astra UI when using Collections API)
```

## Upcoming API changes
- Refactoring builder/factory classes to actually use builder or factory method patterns
- Build abstraction over prompt creation to reduce duplication
- Remove old/unused code
- Refactor dependency injection
- Rewrite graph to build LCEL chains/prompts and deprecate reflection API
- Rewrite tests to use proper mocks,stubs, and fakes 
- Parallelize chains to improve performance
- Consolidate configs
