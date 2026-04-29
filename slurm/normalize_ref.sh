#!/bin/bash


INDEX=$1
if [ -z $INDEX ]; then
    INDEX=0
fi

GPT=gpt-4o-mini

# Non-standard English

CORPUS[0]=rocsmt
REF_FILE[0]=$DATASETS/rocsmt/test/ref.fr.test
TGT_LANG[0]=French

CORPUS[1]=footweets
REF_FILE[1]=$DATASETS/footweets/detok.twitter.sent.de.txt
TGT_LANG[1]=German

# Non-standard French

CORPUS[2]=mmtc
REF_FILE[2]=$DATASETS/mmtc/test.fr-en.en
TGT_LANG[2]=English

CORPUS[3]=pfsmb
REF_FILE[3]=$DATASETS/pfsmb/test.en
TGT_LANG[3]=English

CORPUS[4]=pfsmb-dev
REF_FILE[4]=$DATASETS/pfsmb/dev.en
TGT_LANG[4]=English


echo "Normalizing ${CORPUS[$INDEX]}..."
python $HOME/evaluation-challenges/src/generate_openai.py \
    --input-file ${REF_FILE[$INDEX]} \
    --target-lang ${TGT_LANG[$INDEX]} \
    --model-name $GPT \
    --normalize \
    --guidelines standard
