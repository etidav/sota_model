#!/bin/bash

#
# Script to stop a previously started container nhits
#

container_name="sota_model_${USER}"

cmd="docker rm -f ${container_name}"

echo "will now execute:"
echo $cmd
$cmd

true
