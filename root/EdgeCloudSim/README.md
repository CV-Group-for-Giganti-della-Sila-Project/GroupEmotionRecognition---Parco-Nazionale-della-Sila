This document provides a summary of the Vision-Language Model (VLM) offload simulation scenario files and a step-by-step tutorial on how to configure and run the simulation using EdgeCloudSim.



## 1. Summary of Files

The provided files implement a custom scenario in EdgeCloudSim where fixed cameras periodically send image frames to an Edge computing node running a VLM. The inference results are then sent to Cloud storage. 

### Configuration Files
* **`default_config.properties`**: The master configuration file. It sets the simulation duration (600 minutes), device ranges (1 to 10 cameras), network parameters (WLAN/WAN bandwidth and delays), VM specifications, and selects the `EDGE_VLM_ONLY` orchestrator policy and `PARK_CAMERA_VLM` scenario.
* **`edge_devices.xml`**: Defines the physical layout and specifications of the edge infrastructure. It specifies a single edge datacenter containing a host with a Xen VM (4 cores, 100,000 MIPS, 16GB RAM) to run the VLM inference.
* **`applications.xml`**: Defines the `PARK_FRAME_VLM` application workload. It specifies task sizes (512 KB upload, 5 KB download/result, 200,000 MI task length) and requires 1 core for processing.

### Java Source Files
* **`MainApp.java`**: The entry point for the simulation. It reads the configuration files, initializes the CloudSim engine, runs the simulation loop (iterating through the number of devices), and handles logging output to the `sim_results` directory.
* **`ParkCameraScenarioFactory.java`**: The factory class that binds all custom models together for the `PARK_CAMERA_VLM` scenario, injecting the custom Load Generator, Orchestrator, Mobility Model, and Network Model into the EdgeCloudSim manager.
* **`VlmEdgeOnlyOrchestrator.java`**: A custom edge orchestrator that implements the `EDGE_VLM_ONLY` policy. It forces all tasks to be offloaded to the generic edge device and assigns them to the first available Edge VM.
* **`PeriodicFrameLoadGenerator.java`**: A custom load generator. It simulates fixed cameras sending frames roughly every 20 seconds, with a bounded jitter (±5 seconds) to prevent artificial network synchronization spikes.
* **`FixedCameraMobilityModel.java`**: A custom mobility model ensuring cameras remain stationary. All cameras are assigned the location of the first edge datacenter defined in `edge_devices.xml`.
* **`ParkNetworkModel.java`**: A specialized network model. It calculates upload delays (camera to edge) and download delays. Crucially, it reinterprets the "result download" phase as an edge-to-cloud storage upload for the final VLM results.



## 2. Tutorial: How to Use the Simulation

### Prerequisites
* Java Development Kit (JDK) 8 or higher installed.
* The [EdgeCloudSim](https://github.com/CagataySonmez/EdgeCloudSim) framework cloned or downloaded to your local machine.
* An IDE (like Eclipse or IntelliJ IDEA) or a build tool (Maven/Ant) configured for the EdgeCloudSim project.

### Step 1: Place the Files in the Project
To integrate these files into EdgeCloudSim, place them in the correct directory structure:

1.  **Java Files**: Place `MainApp.java`, `ParkCameraScenarioFactory.java`, `VlmEdgeOnlyOrchestrator.java`, `PeriodicFrameLoadGenerator.java`, `FixedCameraMobilityModel.java`, and `ParkNetworkModel.java` into the following package directory inside your `src` folder:
    `src/main/java/edu/boun/edgecloudsim/applications/vlm_offload/`
2.  **Configuration Files**: Create a directory for the configuration files:
    `scripts/vlm_offload/config/`
    Place `default_config.properties`, `edge_devices.xml`, and `applications.xml` into this folder.

### Step 2: Compile the Project
If you are using an IDE, simply refresh the project and ensure there are no compilation errors. Ensure the CloudSim and EdgeCloudSim core libraries are correctly linked in your build path. 

### Step 3: Run the Simulation
You can run the simulation by executing the `MainApp.java` class.
