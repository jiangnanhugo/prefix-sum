#!/bin/bash

COUNT=0
echo $1;
while [ $COUNT -lt $1 ]; do
    echo $(( RANDOM % 100 ));
    let COUNT=COUNT+1
done
