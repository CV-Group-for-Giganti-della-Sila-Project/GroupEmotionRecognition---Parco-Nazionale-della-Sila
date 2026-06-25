package edu.boun.edgecloudsim.applications.vlm_offload;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * Load generator for the multi-camera VLM scenario.
 *
 * Scenario:
 * - Each mobile device represents one fixed camera.
 * - Every camera sends one frame roughly every 20 seconds.
 * - The interval is not exact; bounded jitter is added.
 * - Initial frame times are randomized slightly so that all cameras do not
 *   send their first frame at the exact same simulation time.
 */
public class PeriodicFrameLoadGenerator extends LoadGeneratorModel {
    private static final int FRAME_TASK_TYPE = 0;

    /*
     * Average frame period.
     * Each camera sends approximately one frame every 20 seconds.
     */
    private static final double BASE_INTERVAL_SECONDS = 20.0;

    /*
     * Bounded jitter.
     * With this value, each interval is between 15 and 25 seconds.
     */
    private static final double MAX_JITTER_SECONDS = 5.0;

    /*
     * Reproducible random generator.
     * Change this seed if you want a different camera trace.
     */
    private static final long RANDOM_SEED = 42L;

    private int[] taskTypeOfDevices;
    private Random random;

    public PeriodicFrameLoadGenerator(
            int numberOfMobileDevices,
            double simulationTime,
            String simScenario) {

        super(numberOfMobileDevices, simulationTime, simScenario);
        this.random = new Random(RANDOM_SEED);
    }

    @Override
    public void initializeModel() {
        taskList = new ArrayList<TaskProperty>();
        taskTypeOfDevices = new int[numberOfMobileDevices];

        /*
         * Every mobile device is a camera.
         * Every camera generates the same task type: PARK_FRAME_VLM.
         */
        for (int deviceId = 0; deviceId < numberOfMobileDevices; deviceId++) {
            taskTypeOfDevices[deviceId] = FRAME_TASK_TYPE;
        }

        /*
         * Read task parameters from applications.xml.
         *
         * Task lookup table indices:
         * [5] data_upload, KB
         * [6] data_download, KB
         * [7] task_length, MI
         * [8] required_core
         */
        double[][] taskLookUpTable = SimSettings.getInstance().getTaskLookUpTable();

        int pesNumber = (int) taskLookUpTable[FRAME_TASK_TYPE][8];
        long taskLengthMi = (long) taskLookUpTable[FRAME_TASK_TYPE][7];

        /*
         * applications.xml stores data sizes in KB.
         * TaskProperty expects bytes.
         */
        long inputFileSizeBytes = (long) (taskLookUpTable[FRAME_TASK_TYPE][5] * 1024L);
        long outputFileSizeBytes = (long) (taskLookUpTable[FRAME_TASK_TYPE][6] * 1024L);

        /*
         * Generate tasks for every camera.
         */
        for (int cameraId = 0; cameraId < numberOfMobileDevices; cameraId++) {
            generateTasksForCamera(
                    cameraId,
                    pesNumber,
                    taskLengthMi,
                    inputFileSizeBytes,
                    outputFileSizeBytes
            );
        }

        /*
         * Sort tasks by start time.
         * This makes the generated workload easier to inspect and keeps the
         * event order clean when multiple cameras are used.
         */
        Collections.sort(taskList, new Comparator<TaskProperty>() {
            @Override
            public int compare(TaskProperty t1, TaskProperty t2) {
                return Double.compare(t1.getStartTime(), t2.getStartTime());
            }
        });
    }

    private void generateTasksForCamera(
            int cameraId,
            int pesNumber,
            long taskLengthMi,
            long inputFileSizeBytes,
            long outputFileSizeBytes) {

        /*
         * Do not start at simulation time 0.
         *
         * Also add a random initial offset so that all cameras do not send
         * frames at exactly the same time.
         */
        double initialOffset = random.nextDouble() * BASE_INTERVAL_SECONDS;

        double virtualTime =
                SimSettings.CLIENT_ACTIVITY_START_TIME
                + 1.0
                + initialOffset;

        while (virtualTime < simulationTime) {
            TaskProperty frameTask = new TaskProperty(
                    virtualTime,              // _startTime
                    cameraId,                 // _mobileDeviceId
                    FRAME_TASK_TYPE,          // _taskType
                    pesNumber,                // _pesNumber
                    taskLengthMi,             // _length
                    inputFileSizeBytes,       // _inputFileSize
                    outputFileSizeBytes       // _outputFileSize
            );

            taskList.add(frameTask);

            double nextInterval = BASE_INTERVAL_SECONDS + getUniformJitter();

            if (nextInterval <= 0) {
                nextInterval = BASE_INTERVAL_SECONDS;
            }

            virtualTime += nextInterval;
        }
    }

    @Override
    public int getTaskTypeOfDevice(int deviceId) {
        if (deviceId < 0 || deviceId >= taskTypeOfDevices.length) {
            throw new IllegalArgumentException(
                    "Invalid deviceId: " + deviceId +
                    ". numberOfMobileDevices = " + taskTypeOfDevices.length
            );
        }

        return taskTypeOfDevices[deviceId];
    }

    private double getUniformJitter() {
        /*
         * Used to simulate delays in the network
         */
        return -MAX_JITTER_SECONDS
                + (2.0 * MAX_JITTER_SECONDS * random.nextDouble());
    }
}