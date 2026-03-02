# CVAT + YOLOv7 Automatic Annotation Setup


This repository explains how to set up CVAT with Serverless Mode (Nuclio) and deploy YOLOv7 for automatic annotation.


At the end of this setup, you will have:


- CVAT running locally


- Nuclio serverless dashboard running


- YOLOv7 selectable inside CVAT’s Automatic Annotation tool


- Automatic annotations to a set of pngs




## Step 1 — Install Prerequisites


### From Ubuntu/Linux:


#### Run the setup script:


`chmod +x scripts/docker_setup.sh`


`bash scripts/docker_setup.sh`


After this you will need to close your ubuntu and reopen it.


### Verify installation:


**`docker --version`**


**`docker compose version`**


**`git --version`**


## Step 2 — Clone CVAT


**`cd ~`**


**`git clone https://github.com/cvat-ai/cvat.git`**


**`cd cvat`**


## Step 3 — Start CVAT


Serverless mode is required for automatic annotation.


**`docker compose -f docker-compose.yml \
  -f components/serverless/docker-compose.serverless.yml up -d`**


Verify containers are running:


**`docker ps | grep nuclio`**


You can verify by seeing a 1 after this:


`docker exec -it cvat_server bash -lc 'echo $CVAT_SERVERLESS'`


## Step 4 — Create Admin User (First Time Only)


**`docker exec -it cvat_server bash -ic 'python3 manage.py createsuperuser'`**


Then open in your local browser:


`http://localhost:8080`


Log in with your new credentials.


### Deploy YOLOv7


Install Nuclio CLI


**`wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64`**


**`sudo mv nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl`**


**`sudo chmod +x /usr/local/bin/nuctl`**


### Deploy the YOLOv7 Function


**`cd ~/cvat`**


#### Then run:


DOCKER_API_VERSION=1.52 nuctl deploy \
  --project-name cvat \
  --path serverless/onnx/WongKinYiu/yolov7/nuclio \
  --file serverless/onnx/WongKinYiu/yolov7/nuclio/function.yaml \
  --platform local \
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
  --platform-config '{"attributes":{"network":"cvat_cvat"}}'


First deployment may take several minutes.


## Verify Deployment


Check function status:


**`DOCKER_API_VERSION=1.52 nuctl get function --platform local`**


You should see:


`STATE: ready`


### Verify serverless mode:


**`docker exec -it cvat_server bash -lc 'echo $CVAT_SERVERLESS'`**


Expected output:


`1`


## Using CVAT


Note: We each will locally run this but can export data into our shared github


Now you can open CVAT in your browser through:


`http://localhost:8080`


### 1. Create a Project


- Go to the Projects tab and click the blue plus sign at the top right


- Create a new project


- Name project and add labels


These labels can include:
- Car


- Truck


- Motorcycle


- Stop sign


### 2. Create a task


Each task will represent a segment of the raw frames.


- Go to the task tab and click the blue plus sign at the top right


- Create a new task


- Name the task (ex: Annotations_Segment_00)


- Select the Project


- Upload the data from the raw file. The file path should look something like this:


`project19/datasets/leaf_run_2/segment_00/raw/`


Note: You need to upload the individual images not the "raw" file


- Click "Submit and Continue"


### 3. Start Annotations


- Inside the task tab, click "Open" on the task you made


- Click the "Actions" button with the three dots


- Click "Automatic Annotations"


- Now you should see "YOLO v7" as one of the models (see Troubleshooting if not)


- Map the tags that you want and click "Annotate"


- After it is done annotating, you can reopen the task and click the blue "job" button at the bottom right


- Now you should be able to go through frame by frame and see the items identified










#### How It Works


CVAT sends images to Nuclio


Nuclio routes the request to the YOLOv7 container


YOLOv7 runs inference


Bounding boxes are returned as JSON


CVAT stores annotations in its database


You view the results directly in the CVAT UI.




## Troubleshooting
### Automatic Annotation Button Missing


Check:


`docker exec -it cvat_server bash -lc 'echo $CVAT_SERVERLESS'`


Must return:


`1`


### YOLOv7 Not Appearing


Check:

'cd ~/cvat'

`DOCKER_API_VERSION=1.52 nuctl get function --platform local`


Must return:


`STATE: ready`

#### If you see "YOLO v7: ERROR/BUILDING" or "Port is already allocated"

Delete the funciton

'DOCKER_API_VERSION=1.52 nuctl delete function onnx-wongkinyiu-yolov7 --platform local'

Redeploy (copy and paste):

DOCKER_API_VERSION=1.52 nuctl deploy \
  --project-name cvat \
  --path serverless/onnx/WongKinYiu/yolov7/nuclio \
  --file serverless/onnx/WongKinYiu/yolov7/nuclio/function.yaml \
  --platform local \
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
  --platform-config '{"attributes":{"network":"cvat_cvat"}}'


### Stop CVAT


`cd ~/cvat`


docker compose -f docker-compose.yml \
  -f components/serverless/docker-compose.serverless.yml down


### Start CVAT Again


`cd ~/cvat`


docker compose -f docker-compose.yml \
  -f components/serverless/docker-compose.serverless.yml up -d
