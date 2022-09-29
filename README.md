# Simple Rule Based IGLU Agent

It's a simple agent for [IGLU Contest](https://www.iglu-contest.net/) that can build stuff from 
fixed position and in empty starting state without using environment information 
if correct resulting grid state is given.

## Prerequisites
- Python 3.7+
- git lfs (for weight download)

Before starting anythong, run the initialize.sh script.

## Restriction of the agent
Agent can work properly if he starts in the empty grid and with provided starting position

## NLP Model
NLP model, used for evaluation can be found [here](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/tree/master/agents/mhb_baseline/nlp_model) 

## How to check agent
You can evaluate agent by starting main.py script, which is using NLP model for target grid generation.

## Broken stuff
Unfortunately flying agent is not working properly, because the flying mode
is not really using continuous space action and one walking step equals to 0.75 of a block length.
Because of that 

## What could be improved
- Finish random start position mode (agent starts at random position)
- Fix flying agent by adding correctional rotating for blocks building
- Add path finding algorithm (A* for example) for walking agent, if 
starting grid state is not empty
- Handle rebuilding of the structures
