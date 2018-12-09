#!/bin/bash 

docker logs -f $(docker run --rm -d --name sentiment -p 3004:3004 sentiment:latest)