package edu.boun.edgecloudsim.applications.vlm_offload;

import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.utils.Location;

/**
 * Network model for the park-camera VLM scenario.
 *
 * Scenario:
 * 1. Camera uploads an image frame to the edge EC2/VLM node.
 * 2. The edge VM runs the VLM inference.
 * 3. The VLM result is sent to cloud storage.
 *
 * Important modeling shortcut:
 * EdgeCloudSim's default SampleMobileDeviceManager expects a "download result"
 * after edge execution. In this scenario, we reinterpret that result-transfer
 * delay as:
 *
 *     edge VLM -> cloud storage
 *
 * Therefore, a task is considered completed after this storage-transfer delay.
 */
public class ParkNetworkModel extends NetworkModel {

    private static final double CAMERA_TO_EDGE_PROPAGATION_DELAY_SECONDS = 0.005;

    private int[] cameraToEdgeClients;
    private int[] edgeToCloudStorageClients;

    public ParkNetworkModel(int numberOfMobileDevices, String simScenario) {
        super(numberOfMobileDevices, simScenario);
    }

    @Override
    public void initialize() {
        int numberOfWlanZones = Math.max(
                1,
                SimSettings.getInstance().getNumOfEdgeDatacenters()
        );

        cameraToEdgeClients = new int[numberOfWlanZones];
        edgeToCloudStorageClients = new int[numberOfWlanZones];
    }

    @Override
    public double getUploadDelay(int sourceDeviceId, int destDeviceId, Task task) {
        Location sourceLocation = SimManager
                .getInstance()
                .getMobilityModel()
                .getLocation(task.getMobileDeviceId(), CloudSim.clock());

        int wlanId = getSafeWlanId(sourceLocation);

        /*
         * Normal path:
         * camera -> edge VLM
         *
         * SampleMobileDeviceManager calls this with:
         * sourceDeviceId = task.getMobileDeviceId()
         * destDeviceId   = nextHopId
         *
         * In EDGE_VLM_ONLY, nextHopId should be GENERIC_EDGE_DEVICE_ID.
         */
        if (destDeviceId == SimSettings.GENERIC_EDGE_DEVICE_ID) {
            return getCameraToEdgeUploadDelay(task.getCloudletFileSize(), wlanId);
        }

        /*
         * Optional path:
         * camera -> cloud
         *
         * Not used by the current EDGE_VLM_ONLY policy, but keeping this
         * makes the model more robust if we later add cloud fallback.
         */
        if (destDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            return getCameraToCloudUploadDelay(task.getCloudletFileSize(), wlanId);
        }

        return -1;
    }

    @Override
    public double getDownloadDelay(int sourceDeviceId, int destDeviceId, Task task) {
        Location destinationLocation = SimManager
                .getInstance()
                .getMobilityModel()
                .getLocation(task.getMobileDeviceId(), CloudSim.clock());

        int wlanId = getSafeWlanId(destinationLocation);

        /*
         * Critical fix:
         *
         * SampleMobileDeviceManager does NOT call:
         *
         *     getDownloadDelay(SimSettings.GENERIC_EDGE_DEVICE_ID, ...)
         *
         * after edge execution.
         *
         * It calls:
         *
         *     getDownloadDelay(task.getAssociatedHostId(), task.getMobileDeviceId(), task)
         *
         * Therefore, sourceDeviceId can be the concrete edge host/datacenter ID,
         * for example 0, rather than GENERIC_EDGE_DEVICE_ID.
         *
         * For this scenario, any non-cloud source is treated as the edge/VLM
         * result source.
         */
        if (isEdgeResultSource(sourceDeviceId)) {
            return getEdgeToCloudStorageDelay(task.getCloudletOutputSize(), wlanId);
        }

        /*
         * Optional future path:
         * cloud storage -> camera/client retrieval.
         */
        if (sourceDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            return getCloudRetrievalDelay(task.getCloudletOutputSize(), wlanId);
        }

        return -1;
    }

    @Override
    public void uploadStarted(Location accessPointLocation, int destDeviceId) {
        int wlanId = getSafeWlanId(accessPointLocation);

        if (destDeviceId == SimSettings.GENERIC_EDGE_DEVICE_ID) {
            cameraToEdgeClients[wlanId]++;
        } else if (destDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            edgeToCloudStorageClients[wlanId]++;
        }
    }

    @Override
    public void uploadFinished(Location accessPointLocation, int destDeviceId) {
        int wlanId = getSafeWlanId(accessPointLocation);

        if (destDeviceId == SimSettings.GENERIC_EDGE_DEVICE_ID) {
            decrement(cameraToEdgeClients, wlanId);
        } else if (destDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            decrement(edgeToCloudStorageClients, wlanId);
        }
    }

    @Override
    public void downloadStarted(Location accessPointLocation, int sourceDeviceId) {
        int wlanId = getSafeWlanId(accessPointLocation);

        /*
         * In the default EdgeCloudSim lifecycle, this callback is called for
         * result delivery. We reinterpret edge result delivery as edge -> cloud
         * storage transfer, so count it on the storage/WAN side.
         */
        if (isEdgeResultSource(sourceDeviceId)) {
            edgeToCloudStorageClients[wlanId]++;
        } else if (sourceDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            edgeToCloudStorageClients[wlanId]++;
        }
    }

    @Override
    public void downloadFinished(Location accessPointLocation, int sourceDeviceId) {
        int wlanId = getSafeWlanId(accessPointLocation);

        if (isEdgeResultSource(sourceDeviceId)) {
            decrement(edgeToCloudStorageClients, wlanId);
        } else if (sourceDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            decrement(edgeToCloudStorageClients, wlanId);
        }
    }

    private boolean isEdgeResultSource(int sourceDeviceId) {
        return sourceDeviceId != SimSettings.CLOUD_DATACENTER_ID;
    }

    private double getCameraToEdgeUploadDelay(long dataSizeBytes, int wlanId) {
        int bandwidthKbps = SimSettings.getInstance().getWlanBandwidth();

        return CAMERA_TO_EDGE_PROPAGATION_DELAY_SECONDS
                + getTransferDelaySeconds(
                        dataSizeBytes,
                        bandwidthKbps,
                        cameraToEdgeClients[wlanId]
                );
    }

    private double getCameraToCloudUploadDelay(long dataSizeBytes, int wlanId) {
        int wlanBandwidthKbps = SimSettings.getInstance().getWlanBandwidth();
        int wanBandwidthKbps = SimSettings.getInstance().getWanBandwidth();

        double wlanDelay =
                CAMERA_TO_EDGE_PROPAGATION_DELAY_SECONDS
                + getTransferDelaySeconds(
                        dataSizeBytes,
                        wlanBandwidthKbps,
                        cameraToEdgeClients[wlanId]
                );

        double wanDelay =
                SimSettings.getInstance().getWanPropagationDelay()
                + getTransferDelaySeconds(
                        dataSizeBytes,
                        wanBandwidthKbps,
                        edgeToCloudStorageClients[wlanId]
                );

        if (wlanDelay < 0 || wanDelay < 0) {
            return -1;
        }

        return wlanDelay + wanDelay;
    }

    private double getEdgeToCloudStorageDelay(long resultSizeBytes, int wlanId) {
        int wanBandwidthKbps = SimSettings.getInstance().getWanBandwidth();

        return SimSettings.getInstance().getWanPropagationDelay()
                + getTransferDelaySeconds(
                        resultSizeBytes,
                        wanBandwidthKbps,
                        edgeToCloudStorageClients[wlanId]
                );
    }

    private double getCloudRetrievalDelay(long resultSizeBytes, int wlanId) {
        int wanBandwidthKbps = SimSettings.getInstance().getWanBandwidth();
        int wlanBandwidthKbps = SimSettings.getInstance().getWlanBandwidth();

        double wanDelay =
                SimSettings.getInstance().getWanPropagationDelay()
                + getTransferDelaySeconds(
                        resultSizeBytes,
                        wanBandwidthKbps,
                        edgeToCloudStorageClients[wlanId]
                );

        double wlanDelay =
                CAMERA_TO_EDGE_PROPAGATION_DELAY_SECONDS
                + getTransferDelaySeconds(
                        resultSizeBytes,
                        wlanBandwidthKbps,
                        cameraToEdgeClients[wlanId]
                );

        if (wanDelay < 0 || wlanDelay < 0) {
            return -1;
        }

        return wanDelay + wlanDelay;
    }

    private double getTransferDelaySeconds(
            long dataSizeBytes,
            int bandwidthKbps,
            int activeClientCount) {

        if (bandwidthKbps <= 0) {
            return -1;
        }

        int clientsSharingTheLink = Math.max(1, activeClientCount + 1);

        double effectiveBandwidthKbps =
                ((double) bandwidthKbps) / clientsSharingTheLink;

        /*
         * Convert bytes to kilobits.
         *
         * bandwidthKbps means kilobits per second.
         */
        double dataSizeKilobits = ((double) dataSizeBytes * 8.0) / 1000.0;

        return dataSizeKilobits / effectiveBandwidthKbps;
    }

    private int getSafeWlanId(Location location) {
        if (location == null) {
            return 0;
        }

        int wlanId = location.getServingWlanId();

        if (wlanId < 0 || wlanId >= cameraToEdgeClients.length) {
            return 0;
        }

        return wlanId;
    }

    private void decrement(int[] array, int index) {
        if (array == null) {
            return;
        }

        if (index >= 0 && index < array.length && array[index] > 0) {
            array[index]--;
        }
    }
}