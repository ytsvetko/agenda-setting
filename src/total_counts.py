#!/usr/bin/env python3

import argparse
import collections
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir')
args = parser.parse_args()

def SaveCounter(counter, filename):
  out_file = open(filename, "w")
  for word, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
    out_file.write(str(count) + " " + word + "\n")
  out_file.close()

def main():
  per_year_counts = collections.defaultdict(collections.Counter)
  for filename in glob.iglob(os.path.join(args.base_dir, "*/*/*.counts")):
    dirname = os.path.dirname(filename)
    year = os.path.basename(dirname)
    print("Processing dir:", dirname)
    counter = per_year_counts[year]
    for line in open(filename):
      count, word = line.split()
      count = int(count)
      counter[word] += count

  total_counter = collections.Counter()
  for year, counter in per_year_counts.items():
    total_counter.update(counter)
    out_filename = os.path.join(args.base_dir, year + ".counts")
    SaveCounter(counter, out_filename)    
  SaveCounter(total_counter, os.path.join(args.base_dir, "total.counts"))

if __name__ == '__main__':
  main()
