#!/bin/bash

mkdir -p /tmp/tokenize

NEWS=$1

for f in ${NEWS} ; do
  echo $f
  base_f=`basename $f`
  tmp_out="/tmp/tokenize/${base_f}"
  ./tokenize-anything.sh < $f  | ./utf8-normalize.sh  > "${tmp_out}"
  ./normalize.py "${tmp_out}" "${f}.tok"
done
