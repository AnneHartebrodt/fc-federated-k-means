#!/bin/bash

docker build . --tag featurecloud.ai/fc-federated-kmeans
docker push featurecloud.ai/fc-federated-kmeans