# Group G Capstone Project19
### Team Members
- Dasanie Le (Sprint Master)
- Trevor Lietch (QA)
- Aleeza Malik (Project Owner)
- John Schruben (QA)
- Blake Shore (QA)

### Project Overview
Our project is to train an existing LLM on dashcam image data so that the LLM can identify relavent elements in dashcam footage and give driving instructions in real-time. This project is LLM Selfdrive for the OU Mobility Intelligence Lab (MiLa).

### Project Objectives
1. Collect dashcam image data using test vehicles at MiLa.
2. Annotate the image data in terms of scene interpretation using ChatGPT and manual correction. For each image, the LLM needs to explain the scene and how the vehicle should drive in order to avoid pedestrians, other vehicles , etc., and to follow traffic laws.
3. Train the lightweight LLMs with the collected data. Run the LLM in real-time with at least 20FPS in test vehicle. (i.e., each BEV space has to be generated within 50ms in a regular computer).
4. Test the trained LLM in vehicle closed course and on-road testing.

### Technologies Used
- Python - programming language for development
- Ollama - open-source tool for running LLMs locally
- GPT-OSS - OpenAI's open-source LLM for local running

### Setup Instructions
1. Download and install the technologies listed above from their sources.
2. Clone the repo to your local machine.

### Progress Plan

