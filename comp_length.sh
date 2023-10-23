#!/usr/bin/env bash

awk '{ if (length($0) > max) max = length($0) } END { print max }' $1

#perl -ne 'chomp; $max = length($_) > $max ? length($_) : $max; END { print "$max\n" }' your_file.txt


