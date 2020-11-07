#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:23:53 2019

@author: alok
"""
# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
reference = [['black', 'dog', 'is', 'running','across','low','cut','field','nearby','an','area','of','trees'],['black','and','white','dog','is','running','through','grassy','field'],['dog','runs','in','the','field']]
candidate = ['dog','is','running','through','the','grass']
print('BLEU-1: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('BLEU-2: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('BLEU-4: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

#1330645772 black dog is running across low cut field nearby an area of trees
#1330645772 large black and white dog is running through grassy field
#1330645772 black and white dog is running in the grass
#1330645772 black dog runs through field
#1330645772 dog runs in field