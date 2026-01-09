#!/bin/bash
docker stop grovio-app 2>/dev/null || true
docker rm grovio-app 2>/dev/null || true
