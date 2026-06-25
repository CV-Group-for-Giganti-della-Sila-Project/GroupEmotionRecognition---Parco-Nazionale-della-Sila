package edu.boun.edgecloudsim.applications.vlm_offload;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

public class MainApp {
    public static final int EXPECTED_NUM_OF_ARGS = 5;

    public static final String APPLICATION_FOLDER = "vlm_offload";

    public static void main(String[] args) {
        Log.disable();
        SimLogger.enablePrintLog();

        int iterationStart;
        int iterationEnd;

        String configFile;
        String outputFolder = null;
        String edgeDevicesFile;
        String applicationsFile;

        if (args.length == EXPECTED_NUM_OF_ARGS) {
            configFile = args[0];
            edgeDevicesFile = args[1];
            applicationsFile = args[2];
            outputFolder = args[3];
            iterationStart = Integer.parseInt(args[4]);
            iterationEnd = iterationStart;
        } else {
            SimLogger.printLine(
                    "Simulation setting file, output folder and iteration number are not provided. " +
                    "Using default ones..."
            );

            configFile = "scripts/" + APPLICATION_FOLDER + "/config/default_config.properties";
            applicationsFile = "scripts/" + APPLICATION_FOLDER + "/config/applications.xml";
            edgeDevicesFile = "scripts/" + APPLICATION_FOLDER + "/config/edge_devices.xml";

            int iteration = 1;
            iterationStart = iteration;
            iterationEnd = iteration;
        }

        SimSettings settings = SimSettings.getInstance();

        if (!settings.initialize(configFile, edgeDevicesFile, applicationsFile)) {
            SimLogger.printLine("Cannot initialize simulation settings!");
            System.exit(0);
        }

        DateFormat dateFormat = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date simulationStartDate = Calendar.getInstance().getTime();

        SimLogger.printLine("Simulation started at " + dateFormat.format(simulationStartDate));
        SimLogger.printLine("----------------------------------------------------------------------");

        for (int iterationNumber = iterationStart; iterationNumber <= iterationEnd; iterationNumber++) {
            if (args.length != EXPECTED_NUM_OF_ARGS) {
                outputFolder = "sim_results/" + APPLICATION_FOLDER + "/ite" + iterationNumber;
            }

            if (settings.getFileLoggingEnabled()) {
                SimLogger.enableFileLog();
                prepareOutputFolder(outputFolder);
            }

            for (int numberOfMobileDevices = settings.getMinNumOfMobileDev();
                 numberOfMobileDevices <= settings.getMaxNumOfMobileDev();
                 numberOfMobileDevices += settings.getMobileDevCounterSize()) {

                for (String simScenario : settings.getSimulationScenarios()) {
                    for (String orchestratorPolicy : settings.getOrchestratorPolicies()) {

                        Date scenarioStartDate = Calendar.getInstance().getTime();

                        SimLogger.printLine("Scenario started at " + dateFormat.format(scenarioStartDate));
                        SimLogger.printLine(
                                "Scenario: " + simScenario +
                                " - Policy: " + orchestratorPolicy +
                                " - #iteration: " + iterationNumber
                        );
                        SimLogger.printLine(
                                "Duration: " + settings.getSimulationTime() / 60 +
                                " min (warm up period: " + settings.getWarmUpPeriod() / 60 +
                                " min) - #devices: " + numberOfMobileDevices
                        );

                        SimLogger.getInstance().simStarted(
                                outputFolder,
                                "SIMRESULT_" + simScenario + "_" +
                                orchestratorPolicy + "_" +
                                numberOfMobileDevices + "DEVICES"
                        );

                        try {
                            int numUser = 2;
                            Calendar calendar = Calendar.getInstance();
                            boolean traceFlag = false;

                            CloudSim.init(numUser, calendar, traceFlag, 0.01);

                            ScenarioFactory scenarioFactory =
                                    new ParkCameraScenarioFactory(
                                            numberOfMobileDevices,
                                            settings.getSimulationTime(),
                                            orchestratorPolicy,
                                            simScenario
                                    );

                            SimManager manager =
                                    new SimManager(
                                            scenarioFactory,
                                            numberOfMobileDevices,
                                            simScenario,
                                            orchestratorPolicy
                                    );

                            manager.startSimulation();
                        } catch (Exception e) {
                            SimLogger.printLine(
                                    "The simulation has been terminated due to an unexpected error."
                            );
                            e.printStackTrace();
                            System.exit(0);
                        }

                        Date scenarioEndDate = Calendar.getInstance().getTime();

                        SimLogger.printLine(
                                "Scenario finished at " +
                                dateFormat.format(scenarioEndDate) +
                                ". It took " +
                                SimUtils.getTimeDifference(scenarioStartDate, scenarioEndDate)
                        );
                        SimLogger.printLine("----------------------------------------------------------------------");
                    }
                }
            }
        }

        Date simulationEndDate = Calendar.getInstance().getTime();

        SimLogger.printLine(
                "Simulation finished at " +
                dateFormat.format(simulationEndDate) +
                ". It took " +
                SimUtils.getTimeDifference(simulationStartDate, simulationEndDate)
        );
    }

    private static void prepareOutputFolder(String outputFolder) {
        File dir = new File(outputFolder);

        if (dir.exists() && dir.isDirectory()) {
            SimLogger.printLine("Output folder is available; cleaning '" + outputFolder + "'");

            File[] files = dir.listFiles();

            if (files != null) {
                for (File file : files) {
                    if (file.exists() && file.isFile()) {
                        if (!file.delete()) {
                            SimLogger.printLine("File cannot be deleted: " + file.getAbsolutePath());
                            System.exit(1);
                        }
                    }
                }
            }
        } else {
            SimLogger.printLine("Output folder is not available; creating '" + outputFolder + "'");
            if (!dir.mkdirs()) {
                SimLogger.printLine("Output folder could not be created: " + outputFolder);
                System.exit(1);
            }
        }
    }
}