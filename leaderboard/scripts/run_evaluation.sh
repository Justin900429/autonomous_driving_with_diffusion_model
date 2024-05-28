#!/bin/bash
export CARLA_ROOT=/home/bokai/Documents/Carla_9.10
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
echo ${PYTHONPATH}

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True


# evaluation
export ROUTES=leaderboard/data/evaluation_routes/routes_town05_long.xml # DO NOT CHANGE THIS
export TEAM_AGENT=e2e_driving/vision_agent.py 
export TEAM_CONFIG=log/VA/best_epoch=56-val_loss=0.298.ckpt
export CHECKPOINT_ENDPOINT=results_longest_5.json 
export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json # DO NOT CHANGE THIS
export SAVE_PATH=data/results_longest_5/ 


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}


