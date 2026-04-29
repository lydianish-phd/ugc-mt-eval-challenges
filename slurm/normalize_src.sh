#!/bin/bash

INDEX=$1
if [ -z $INDEX ]; then
    INDEX=0
fi

GPT=gpt-4o-mini

# Non-standard English

CORPUS[0]=rocsmt
SRC_FILE[0]=$DATASETS/rocsmt/test/raw.en.test
SRC_LANG[0]=English

CORPUS[1]=footweets
SRC_FILE[1]=$DATASETS/footweets/detok.twitter.sent.en.txt
SRC_LANG[1]=English

CORPUS[2]=mtnt
SRC_FILE[2]=$DATASETS/mtnt/MTNT2019/en-fr.en
SRC_LANG[2]=English

CORPUS[3]=mtnt
SRC_FILE[3]=$DATASETS/mtnt/MTNT2019/en-ja.en
SRC_LANG[3]=English

# Non-standard French

CORPUS[4]=foursquare
SRC_FILE[4]=$DATASETS/foursquare/test.fr
SRC_LANG[4]=French

CORPUS[5]=mtnt
SRC_FILE[5]=$DATASETS/mtnt/MTNT2019/fr-en.fr
SRC_LANG[5]=French

CORPUS[6]=mmtc
SRC_FILE[6]=$DATASETS/mmtc/test.fr-en.fr
SRC_LANG[6]=French

CORPUS[7]=pfsmb
SRC_FILE[7]=$DATASETS/pfsmb/test.fr
SRC_LANG[7]=French

CORPUS[8]=pfsmb-dev
SRC_FILE[8]=$DATASETS/pfsmb/dev.fr
SRC_LANG[8]=French


echo "Normalizing ${CORPUS[$INDEX]}..."
python $HOME/evaluation-challenges/src/generate_openai.py \
    --input-file ${SRC_FILE[$INDEX]} \
    --target-lang ${SRC_LANG[$INDEX]} \
    --model-name $GPT \
    --normalize