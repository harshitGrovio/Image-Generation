#!/bin/bash
cd /home/ec2-user/app
pip3 install --no-cache-dir -r requirements.txt --quiet &
disown
