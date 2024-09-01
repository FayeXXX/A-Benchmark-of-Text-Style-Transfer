#!/usr/bin/env bash
bl=(0.1 0.2 0.3)
sc=(1.0 1.2 1.5)

for s in "${sc[@]}"
do
  for b in "${bl[@]}"
  do
    bash pg_arr.sh ARR 0 ${s} ${b} sc bl
  done
done