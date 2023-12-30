# Moonshot
![moonshot_graphic.png](img%2Fmoonshot_graphic.png)

## Automatic Hyperpersonalization

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
PYTHONUNBUFFERED=1;
ASTRA_DB_TOKEN_BASED_PASSWORD=AstraCS:...;
IBM_API_SECRET=...;
IBM_PROJECT_ID=...;
KEYSPACE=openai;
OPENAI_API_KEY=...;
PROVIDER=OPENAI;
TABLE_NAME=mytable;
```
