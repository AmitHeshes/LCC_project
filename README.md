# LCC_project
*Install kenlm using pip install after dealing with dependings (using linux, practically wsl).
*Create the train text data (text file) using function create_train_file in create_train_file.
*Move the train file to linux.
*Run in bash (linux) the following line (with parameter change - name of arpa is model_3gram.arpa): ./bin/lmplz -o 3 < ../text.txt > text.arpa
*Run in bash (linux) the following line (with parameter change - name of binary is model_3gram.binary): ./bin/build_binary text.arpa text.binary
*Move the arpa and binary model files to git directory
*Run surprisal_sanity to check the pipline worked  


Added two variations of pythia surprisals - one which gives average surprisals of the tokens consisting each word (according to pythia 70M tokenizer), and the other summing over these tokens (like multiply probabilities).
