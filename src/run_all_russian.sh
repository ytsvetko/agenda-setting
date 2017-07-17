#!/bin/bash

BASE_DIR=/usr1/home/ytsvetko/projects/agenda_setting
_LANGUAGE_=russian

NEWS=${BASE_DIR}/data/news/${_LANGUAGE_}

## Crawl the articles
# ${BASE_DIR}/src/crawlers/${_LANGUAGE_}/crawl_eastview.py

## Convert HTML to plain text
${BASE_DIR}/src/crawlers/${_LANGUAGE_}/html_to_text.py --base_dir ${NEWS} --overwrite

## Tokenize
cd ${BASE_DIR}/src/tokenizer 
 ./tokenize.sh "${NEWS}/*/*/*.txt"
cd -

# Count tokens per dir
#${BASE_DIR}/src/per_year_counts.py --base_dir ${NEWS}

# Summarize counts among all sources
#${BASE_DIR}/src/total_counts.py --base_dir ${NEWS}


### TODO 1
## Train LDA model
LDA_MODEL="${BASE_DIR}/models/lda_russian.pkl"
${BASE_DIR}/src/lda.py --article_glob "${NEWS}/*/*/*.txt.tok" \
    --force_train --output_topic_distribution --lda_model ${LDA_MODEL}
#    --stopwords ${BASE_DIR}/data/stopwords/${_LANGUAGE_}_stopwords.txt \


### TODO 2
## Get topic distribution of an external-policy related article using trained LDA model
${BASE_DIR}/src/lda.py \
    --article_glob "${BASE_DIR}/data/focused_crawl/${_LANGUAGE_}/*.txt.tok" \
    --output_topic_distribution --lda_model ${LDA_MODEL}
#    --stopwords ${BASE_DIR}/data/stopwords/${_LANGUAGE_}_stopwords.txt \


# Print number of articles about external policy per year per newspaper
MONTHS=(1 2 3 4 5 6 7 8 9 10 11 12)
NEWSPAPERS=(Izvestiia Pravda)

for newspaper in ${NEWSPAPERS[@]}; do 
  log_file=../work/topic_stats_${newspaper}.lda
  touch ${log_file}
  for month in ${MONTHS[@]}; do 
    echo "Processing ${newspaper}"
    ${BASE_DIR}/src/per_month_lda.py \
      --gold_vectors "${BASE_DIR}/data/focused_crawl/${_LANGUAGE_}/external_policy.txt.tok.lda" \
      --log_file "${log_file}" \
      --per_month_glob "${NEWS}/${newspaper}/*/*_${month}.txt.tok.lda"
  done
done 

# Plot GDP, internal, external

