#! /bin/bash

docker run -it --gpus all --shm-size=16G \
--name nia-test \
-v /hosted/workspace/1_user/dkswns333@agilesoda.ai/NIA/nia-82-134:/workspace \
-v /hosted/workspace/2_public_data/NIA2021:/data \
nia-82-134:final
