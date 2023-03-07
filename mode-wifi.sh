#!/bin/bash
sudo systemctl unmask wpa_supplicant
sudo systemctl start wpa_supplicant
sudo systemctl start NetworkManager

#WifiをGUIで接続
