#!/bin/bash
#wifiをGUIで切断

#固定IP設定 $1のところのIPは　任意 コマンドライン引数 (ex. 192.168.1.1)
sudo ifconfig wlan0 "$1" netmask 255.255.255.0
sudo systemctl stop NetworkManager
sudo systemctl mask wpa_supplicant
sudo ifconfig wlan0 down
sudo iwconfig wlan0 mode ad-hoc essid test
sudo ip link set wlan0 up
sudo olsrd -i wlan0 -d 1

#神
