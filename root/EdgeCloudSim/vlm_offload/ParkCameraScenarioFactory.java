package edu.boun.edgecloudsim.applications.vlm_offload;

import edu.boun.edgecloudsim.applications.tutorial1.SampleMobileDeviceManager;
import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.cloud_server.DefaultCloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.DefaultMobileServerManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.DefaultEdgeServerManager;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;

public class ParkCameraScenarioFactory implements ScenarioFactory {
    private final int numOfMobileDevice;
    private final double simulationTime;
    private final String orchestratorPolicy;
    private final String simScenario;

    public ParkCameraScenarioFactory(
            int numOfMobileDevice,
            double simulationTime,
            String orchestratorPolicy,
            String simScenario) {

        this.numOfMobileDevice = numOfMobileDevice;
        this.simulationTime = simulationTime;
        this.orchestratorPolicy = orchestratorPolicy;
        this.simScenario = simScenario;
    }

    @Override
    public LoadGeneratorModel getLoadGeneratorModel() {

        return new PeriodicFrameLoadGenerator(
                numOfMobileDevice,
                simulationTime,
                simScenario
        );
    }

    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        /*
         * Always sends the camera frame task to the edge VM,
         * which represents the EC2 instance running the VLM.
         */
        return new VlmEdgeOnlyOrchestrator(orchestratorPolicy, simScenario);
    }

    @Override
    public MobilityModel getMobilityModel() {
        /*
         * Static cameras inside the park.
         */
        return new FixedCameraMobilityModel(numOfMobileDevice, simulationTime);
    }

    @Override
    public NetworkModel getNetworkModel() {
        /*
         * Models:
         * 1. camera -> edge frame upload
         * 2. edge -> cloud storage upload for the VLM result
         */
        return new ParkNetworkModel(numOfMobileDevice, simScenario);
    }

    @Override
    public EdgeServerManager getEdgeServerManager() {
  
        return new DefaultEdgeServerManager();
    }

    @Override
    public CloudServerManager getCloudServerManager() {
        return new DefaultCloudServerManager();
    }

    @Override
    public MobileServerManager getMobileServerManager() {
        return new DefaultMobileServerManager();
    }

    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        return new SampleMobileDeviceManager();
    }
}