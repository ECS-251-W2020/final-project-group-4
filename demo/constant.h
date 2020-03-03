#pragma once
const int AES_NR = 10; // numbers of rounds
const int AES_NK = 4;  // numbers of columns in a key
const int AES_NB = AES_NK * 4; // key size
const int AES_EXP_NB = (AES_NR + 1) * AES_NB; // expanded key size