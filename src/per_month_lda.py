#!/usr/bin/env python3

import argparse
import collections
import os
import glob
from scipy.stats.stats import pearsonr
from scipy import spatial
import math 

parser = argparse.ArgumentParser()
parser.add_argument('--gold_vectors', 
    default="../focused_crawl/external_policy.txt.tok.lda")
parser.add_argument('--per_month_glob',
    default="../news/russian/Izvestiia/*/*.txt.tok.lda")
parser.add_argument('--similarity_threshold',
    type=float, default=0.4)
parser.add_argument('--log_file',
    default="../work/topic_stats.lda")    
args = parser.parse_args()

NEW_ARTICLE_TOKEN="NEW - ARTICLE - TOKEN"

def ArticleIter(filename):
  current_article = []
  for line in open(filename):
    line = line.strip()
    if not line:
      continue
    if line == NEW_ARTICLE_TOKEN:
      if current_article:
        yield "\n".join(current_article)
        current_article = []
    else:
      current_article.append(line)
  if current_article:
    yield "\n".join(current_article)

def LoadArticles(filename):
  articles = []
  article_index = []
  for index, article in enumerate(ArticleIter(filename)):
    articles.append(article)
    article_index.append((filename, index))
  return articles


def Similarity(v1, v2, metric="cosine"):
  def IsZero(v):
    return all(n == 0 for n in v)    

  if metric == "correlation":
    if IsZero(v1) or IsZero(v2):
      return 0.0
    return pearsonr(v1, v2)[0]

  if metric == "abs_correlation":
    if IsZero(v1) or IsZero(v2):
      return 0.0
    return abs(pearsonr(v1, v2)[0])

  if metric == "cosine":
    return spatial.distance.cosine(v1, v2)

def LoadVectors(filename):
  vectors = []
  for line in open(filename):
    vector = [float(n) for n in line.split()]
    if len(vector) == 0:
      continue
    # normalize
    sqrt_length = math.sqrt(sum([n**2 for n in vector]) + 1e-6)
    vectors.append([n/sqrt_length for n in vector])
  return vectors
  

def GetSimilarArticles(articles, vectors, gold_vector, threshold):
  similar = []
  for article, vector in zip(articles, vectors):
    if Similarity(vector, gold_vector) < threshold:
      similar.append(article)
  return similar
      
 
def main():
  out_f = open(args.log_file, "a")
  gold_vector = LoadVectors(args.gold_vectors)[0]
  out_f.write(args.per_month_glob)
  out_f.write("\n")
  #stats = []
  for filename in sorted(glob.iglob(args.per_month_glob)):
    dirname = os.path.dirname(filename)
    year_month = os.path.basename(filename.replace(".txt.tok.lda", ""))
    vectors = LoadVectors(filename)
    articles = LoadArticles(filename.replace(".lda", ""))
    similar_articles = GetSimilarArticles(articles, vectors, gold_vector, 
        args.similarity_threshold)
    out_f.write("{}\t{}\t{}\n".format(year_month, len(similar_articles), len(vectors)))

    
      
if __name__ == '__main__':
  main()
