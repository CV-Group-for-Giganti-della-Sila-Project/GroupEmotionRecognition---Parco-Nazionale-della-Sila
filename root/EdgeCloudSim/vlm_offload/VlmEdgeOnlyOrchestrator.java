package edu.boun.edgecloudsim.applications.vlm_offload;

import java.util.List;

import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.utils.SimLogger;

/**
 * Edge-only orchestrator for the VLM camera scenario.
 *
 * Scenario:
 * - The camera sends frames.
 * - The VLM is deployed on the edge EC2-like instance.
 * - Therefore, all inference tasks should execute on an edge VM.
 */
public class VlmEdgeOnlyOrchestrator extends EdgeOrchestrator {
    public static final String POLICY_EDGE_VLM_ONLY = "EDGE_VLM_ONLY";

    private int numberOfHosts;

    public VlmEdgeOnlyOrchestrator(String policy, String simScenario) {
        super(policy, simScenario);
    }

    @Override
    public void initialize() {
        numberOfHosts = SimSettings.getInstance().getNumOfEdgeHosts();

        if (numberOfHosts <= 0) {
            SimLogger.printLine(
                    "VlmEdgeOnlyOrchestrator error: no edge hosts found. " +
                    "Check scripts/vlm_offload/config/edge_devices.xml."
            );
        }
    }

    @Override
    public int getDeviceToOffload(Task task) {
        /*
         * In this scenario the VLM is on the edge.
         * Therefore every frame goes to the generic edge device.
         */
        if (policy.equalsIgnoreCase(POLICY_EDGE_VLM_ONLY)) {
            return SimSettings.GENERIC_EDGE_DEVICE_ID;
        }

        SimLogger.printLine(
                "Warning: unknown policy '" + policy +
                "'. Falling back to EDGE_VLM_ONLY."
        );

        return SimSettings.GENERIC_EDGE_DEVICE_ID;
    }

    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        if (deviceId != SimSettings.GENERIC_EDGE_DEVICE_ID) {
            SimLogger.printLine(
                    "VlmEdgeOnlyOrchestrator error: unsupported deviceId " +
                    deviceId + ". This scenario only supports edge execution."
            );
            return null;
        }

        /*
         * Scan all edge hosts and return the first VM that is already assigned
         * to a host.
         */
        for (int hostIndex = 0; hostIndex < numberOfHosts; hostIndex++) {
            List vmList = SimManager
                    .getInstance()
                    .getEdgeServerManager()
                    .getVmList(hostIndex);

            if (vmList == null || vmList.isEmpty()) {
                continue;
            }

            for (Object vmObject : vmList) {
                Vm vm = (Vm) vmObject;
                if (vm.getHost() != null) {
                    return vm;
                }
            }
        }

        /*
         * If we are here, either:
         * 1. the first task was scheduled too early, or
         * 2. the VM could not be allocated to the host, or
         * 3. edge_devices.xml defines no valid edge VM.
         *
         * The first problem is handled by delaying the first frame in
         * PeriodicFrameLoadGenerator.
         */
        SimLogger.printLine(
                "No allocated edge VM found for task at simulation time. " +
                "Possible causes: first task scheduled too early, edge VM " +
                "does not fit into host resources, or edge_devices.xml has " +
                "no valid VM."
        );

        return null;
    }

    @Override
    public void processEvent(SimEvent event) {
        /*
         * This orchestrator does not process custom events.
         */
    }

    @Override
    public void shutdownEntity() {
        /*
         * Nothing to clean up.
         */
    }

    @Override
    public void startEntity() {
        /*
         * Nothing to schedule at startup.
         */
    }
}