# Group G Capstone Project19
### Team Members
- Dasanie Le (Sprint Master)
- Trevor Lietch (QA)
- Aleeza Malik (Project Owner)
- John Schruben (QA)
- Blake Shore (QA)
---
### Project Overview
Main Task: Our project is to train an existing LLM on dashcam image data so that the LLM can identify elements in dashcam footage and give driving instructions in response to the footage. This project is LLM Selfdrive for the OU Mobility Intelligence Lab (MiLa).
Side Task: We will program an automatic data preparation pipeline for autonomous driving. 

### Main Task Objectives
1. Collect dashcam image data using test vehicles at MiLa.
2. Annotate the image data in terms of scene interpretation using ChatGPT and manual correction. For each image, the LLM needs to explain the scene and how the vehicle should drive in order to avoid pedestrians, other vehicles, etc., and to follow traffic laws.
3. Train the lightweight LLMs with the collected data. Run the LLM in real-time with at least 20FPS in test vehicle. (i.e., each BEV space has to be generated within 50ms in a regular computer).
4. Test the trained LLM in vehicle closed course and on-road testing.

### Side Task Objectives
1. Data collection: Integrate the MiLa repo to automatically generate images and some features (features will be used in later E2E learning training) based on MiLa test vehicle dashcam data collected on Comma 3x device: https://github.com/OUMiLa/Openpilot_Custom/tree/main
2. Data automatic annotation: Integrate CVAT opensource code to achieve automatic image annotation for object detection (pedestrian, vehicle, traffic light, stop sign, road boundary, traffic signs, etc.), segmentation, depth estimation, etc. Modify CVAT codebase and customize it to fit our needs as CVAT is mainly for object detection.
3. Data label manual correction: Allow human experts to check the automatic annotated labels and manually correct them using the mouse if necessary.
4. Database (Labelled data storage): Build a database and integrated into your program to store the annotated data in step 3. In this database, it should be conveniently to save and extract certain data and labels.
5. Data export for training (annotated data filtering based on different training needs): Allow user to easily filter the data of interest based on labels.
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
- Research best LLM for our project.
- Get the local LLM running on our computers.
- **Side**: Integrate the MiLa repo to automatically generate images based on MiLa test vehicle dashcam data collected.
- **Side**: Integrate CVAT opensource code to achieve automatic image annotation for object detection.

Sprint 2
- Build a program for training the LLM on detecting necessary elements in the dashcam images.
- Train the LLM to detect the elements. It should output a description of the scene.
- Test LLM's ability to detect elements correctly.
- **Side**: Build manual correction capability for annotated dashcam images.

Sprint 3
- Build a program for training the LLM on making correct driving instructions.
- Train the LLM to make driving instructions in response to dashcam images. It should output a trajectory of where to go and the accelerator, brake, and steering actions needed. (There is no specific format needed.)
- Test LLM's ability to give correct driving instructions.
- **Side**: Build a database and integrate it to store the annotated dashcam images. Build data export for training.

Sprint 4
- Build a program for running the trained LLM with live dashcam footage from the MiLa vehicles.
- Test the latency of the LLM output, and optimize code to reduce it.
- Test the LLM on the MiLa vehicles in closed course and on-road testing.
