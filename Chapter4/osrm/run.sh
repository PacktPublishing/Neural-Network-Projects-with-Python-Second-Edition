#!/usr/bin/env bash

FLAG_FILE=us-northeast-latest.processed

if [ ! -f $FLAG_FILE ]; then
  if [ ! -f us-northeast-latest.osm.pbf ]; then
#    wget https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
    wget https://download.geofabrik.de/north-america/us-northeast-latest.osm.pbf
  fi

  docker pull osrm/osrm-backend

  docker run -t -v $(pwd):/data osrm/osrm-backend \
      osrm-extract -p /opt/car.lua /data/us-northeast-latest.osm.pbf

  docker run -t -v $(pwd):/data osrm/osrm-backend \
      osrm-partition /data/us-northeast-latest.osrm

  docker run -t -v $(pwd):/data osrm/osrm-backend \
      osrm-customize /data/us-northeast-latest.osrm
  touch $FLAG_FILE
fi

docker run -p 5050:5000 \
    -v $(pwd):/data \
    osrm/osrm-backend \
    osrm-routed --algorithm mld -l WARNING /data/us-northeast-latest.osrm
