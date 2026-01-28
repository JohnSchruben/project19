# Group G Capstone Project19
### Team Members
- Dasanie Le (Sprint Master)
- Trevor Lietch (QA)
- Aleeza Malik (Project Owner)
- John Schruben (QA)
- Blake Shore (QA)
---
### Project Overview
Our project is to train an existing LLM on dashcam image data so that the LLM can identify relavent elements in dashcam footage and give driving instructions in real-time. This project is LLM Selfdrive for the OU Mobility Intelligence Lab (MiLa).

### Project Objectives
1. Collect dashcam image data using test vehicles at MiLa.
2. Annotate the image data in terms of scene interpretation using ChatGPT and manual correction. For each image, the LLM needs to explain the scene and how the vehicle should drive in order to avoid pedestrians, other vehicles, etc., and to follow traffic laws.
3. Train the lightweight LLMs with the collected data. Run the LLM in real-time with at least 20FPS in test vehicle. (i.e., each BEV space has to be generated within 50ms in a regular computer).
4. Test the trained LLM in vehicle closed course and on-road testing.
---
### Technologies Used
- Python - programming language for development
- Ollama - open-source tool for running LLMs locally
- GPT-OSS - OpenAI's open-source LLM for local running

### Setup Instructions
1. Download and install the technologies listed above from their sources.
2. Clone the repo to your local machine.
---
### Roadmap
Sprint 1
- Meet with mentor to receive technical info about how the MiLa self-driving cars work so we know how to program for them. Document info and update roadmap as necessary.
- Get a local LLM running on our computers.
- Get dashcam image data using the MiLa test vehicles.
- Annotate the dashcam image data for scene interpretation.

Sprint 2
- Build a program for training the LLM on detecting necessary elements in the dashcam images.
- Train the LLM to detect the elements.
- Test LLM's ability to detect elements correctly.

Sprint 3
- Build a program for training the LLM on making correct driving instructions as needed by the MiLa vehicles.
- Train the LLM to make driving instructions in response to dashcam images.
- Test LLM's ability to give correct driving instructions.

Sprint 4
- Build a program for running the trained LLM with live dashcam footage from the MiLa vehicles.
- Test the latency of the LLM output, and optimize code to reduce it.
- Test the LLM on the MiLa vehicles in closed course and on-road testing.
