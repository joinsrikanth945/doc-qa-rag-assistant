# Nimbus Hub — Network and Connectivity Guide

*Fictional product documentation, written for evaluating this project.*

## Wi-Fi requirements

The Nimbus Hub connects only to 2.4 GHz Wi-Fi networks using WPA2 or WPA3 security. Networks with a captive portal, such as hotel or campus Wi-Fi, are not supported. If your router broadcasts a combined 2.4/5 GHz network under one name, temporarily disable the 5 GHz band during setup.

## Pairing devices

Press the pairing button on the back of the hub once; the ring light pulses green for 120 seconds. Within that window, open the Nimbus app, tap "Add device" and follow the instructions for your device. The hub can pair up to 64 devices, including thermostats, sensors and smart plugs, using the Zigbee 3.0 protocol.

## Static IP and ports

Most homes do not need any network changes. If your network uses strict firewall rules, allow outbound TCP traffic on port 8883 (MQTT over TLS) and port 443 to cloud.nimbus.example. You can reserve a static IP address for the hub in your router's DHCP settings; the hub's MAC address is printed on the label under the stand.

## Offline mode

If the internet connection drops, the hub keeps running schedules and automations locally for up to 30 days. Remote control through the app and voice assistants is unavailable until the connection returns. Events recorded while offline are uploaded automatically once the hub reconnects.

## Status light

| Light | Meaning |
|-------|---------|
| Solid white | Connected and working normally |
| Pulsing green | Pairing mode is active |
| Solid amber | No internet connection; running in offline mode |
| Blinking red | Hardware fault; unplug the hub for 10 seconds, then plug it back in |

## Resetting the hub's network settings

Hold the pairing button for 10 seconds until the ring light flashes amber three times. This clears the stored Wi-Fi network but keeps paired devices. To erase paired devices as well, perform a full reset by holding the button for 30 seconds.
