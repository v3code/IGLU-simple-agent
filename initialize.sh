#!/bin/sh
pip install -r requirements.txt

if [ "$1" = "--colab" ]; then
    git lfs clone http://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline.git
else
    git clone http://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-rl-mhb-baseline.git
fi

cp -r ./iglu-2022-rl-mhb-baseline/agents/mhb_baseline/nlp_model/model ./nlp_model/
cp -r ./iglu-2022-rl-mhb-baseline/agents/mhb_baseline/nlp_model/tokenizer ./nlp_model/
cp ./iglu-2022-rl-mhb-baseline/agents/mhb_baseline/nlp_model/t5-autoregressive-history-3-best.pt ./nlp_model/
rm -rf ./iglu-2022-rl-mhb-baseline