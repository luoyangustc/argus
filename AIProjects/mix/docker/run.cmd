#!/bin/bash

mkdir -p /tmp/logs

PORT_HTTP=23401 ./serving-eval -f serving-eval.conf 2>/tmp/logs/1 &
sleep 3
sleep 5
PORT_HTTP=23400 ./mix -f mix.conf