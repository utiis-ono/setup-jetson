#!/bin/bash

for i in {0..10}
do
	python3 my_client.py move 900 2>&1 | tee -a worker2_log.csv
	echo $((i++));
	for j in {0..5}
	do
		python3 my_client.py move -1 2>&1 | tee -a worker2_log.csv
		echo $((j++));
	done
done
