#!/bin/sh
pip install -r requirements.txt
mkdir ./nlp_model/model
mkdir ./nlp_model/tokenizer
wget https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/raw/master/agents/mhb_baseline/nlp_model/t5-autoregressive-history-3-best.pt -P ./nlp_model/
wget https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/raw/master/agents/mhb_baseline/nlp_model/model/config.json -P ./nlp_model/model/
wget https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/raw/master/agents/mhb_baseline/nlp_model/model/pytorch_model.bin -P ./nlp_model/model/
wget https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/raw/master/agents/mhb_baseline/nlp_model/model/vocab.txt -P ./nlp_model/model/
wget https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/raw/master/agents/mhb_baseline/nlp_model/tokenizer/special_tokens_map.json -C ./nlp_model/tokenizer/
wget https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/raw/master/agents/mhb_baseline/nlp_model/tokenizer/spiece.model -P ./nlp_model/tokenizer/
wget https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/raw/master/agents/mhb_baseline/nlp_model/tokenizer/tokenizer_config.json -P ./nlp_model/tokenizer/
wget https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline/-/raw/master/agents/mhb_baseline/nlp_model/tokenizer/vocab.txt -P ./nlp_model/tokenizer/
echo "Initialization finished"