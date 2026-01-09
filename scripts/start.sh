#!/bin/bash
cd /home/ec2-user/app
nohup python3 api_server.py > /home/ec2-user/app.log 2>&1 &
