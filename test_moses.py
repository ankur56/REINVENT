#!/usr/bin/env python3

import moses


with open('train_pas.smi', 'r') as file:
    train_list = [line.strip() for line in file.readlines()]

with open('gen_pas.smi', 'r') as file:
    gen_list = [line.strip() for line in file.readlines()]

metrics = moses.get_all_metrics(gen_list, k=None, n_jobs=16,
                                device='cpu', batch_size=512, pool=None,
                                test=train_list, test_scaffolds=train_list,
                                ptest=None, ptest_scaffolds=None,train=train_list)

print(metrics)

