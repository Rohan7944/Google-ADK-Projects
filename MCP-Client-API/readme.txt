This section assumes that already have python installed.

Step1 - Clone the repo using - git clone <repo_url>

Step2 - Create a venv environment using - python -m venv .venv and then activate it using - .\venv\Scripts\activate

Step3 - Install the dependencies that are individually mentioned in "requirements.txt" file under every folder(API,Cred-API,MCP-Client,MCP-Server)
(You can also create individual venv environment and install dependencies for it if you want to run a specific server only)

Step4 - Run API server using - python main.py

Step5 - Run Cred-API server using - python main.py

Step6 - Run MCP-Server using - python main.py

Step7 - Run MCP-Client using - python main.py