package edu.boun.edgecloudsim.applications.vlm_offload;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.utils.Location;


public class FixedCameraMobilityModel extends MobilityModel {
    private Location[] deviceLocations;

    public FixedCameraMobilityModel(int numberOfMobileDevices, double simulationTime) {
        super(numberOfMobileDevices, simulationTime);
    }

    @Override
    public void initialize() {
        deviceLocations = new Location[numberOfMobileDevices];

        /*
         * Read the first edge datacenter location from edge_devices.xml.
         *
         * This keeps the camera's location consistent with the WLAN ID,
         * x position, y position, and attractiveness/place type configured
         * in the edge_devices.xml file.
         */
        Document edgeDevicesDocument =
                SimSettings.getInstance().getEdgeDevicesDocument();

        NodeList datacenterList =
                edgeDevicesDocument.getElementsByTagName("datacenter");

        if (datacenterList == null || datacenterList.getLength() == 0) {
            throw new IllegalStateException(
                    "No datacenter found in edge_devices.xml. " +
                    "FixedCameraMobilityModel needs at least one edge datacenter."
            );
        }

        Element datacenterElement = (Element) datacenterList.item(0);

        NodeList locationList =
                datacenterElement.getElementsByTagName("location");

        if (locationList == null || locationList.getLength() == 0) {
            throw new IllegalStateException(
                    "No location found for the first datacenter in edge_devices.xml."
            );
        }

        Element locationElement = (Element) locationList.item(0);

        int placeTypeIndex = readIntElement(locationElement, "attractiveness", 0);
        int servingWlanId = readIntElement(locationElement, "wlan_id", 0);
        int xPos = readIntElement(locationElement, "x_pos", 0);
        int yPos = readIntElement(locationElement, "y_pos", 0);

        Location fixedCameraLocation =
                new Location(placeTypeIndex, servingWlanId, xPos, yPos);

        /*
         * Assign the same fixed location to all devices.
         */
        for (int deviceId = 0; deviceId < numberOfMobileDevices; deviceId++) {
            deviceLocations[deviceId] = fixedCameraLocation;
        }
    }

    @Override
    public Location getLocation(int deviceId, double time) {
        if (deviceLocations == null) {
            throw new IllegalStateException(
                    "FixedCameraMobilityModel has not been initialized yet."
            );
        }

        if (deviceId < 0 || deviceId >= numberOfMobileDevices) {
            throw new IllegalArgumentException(
                    "Invalid deviceId: " + deviceId +
                    ". numberOfMobileDevices = " + numberOfMobileDevices
            );
        }

        return deviceLocations[deviceId];
    }

    private int readIntElement(Element parent, String tagName, int defaultValue) {
        NodeList nodes = parent.getElementsByTagName(tagName);

        if (nodes == null || nodes.getLength() == 0) {
            return defaultValue;
        }

        String text = nodes.item(0).getTextContent();

        if (text == null || text.trim().isEmpty()) {
            return defaultValue;
        }

        return Integer.parseInt(text.trim());
    }
}