# EKERC
CODE FOR EKERC
Data DIR Structure
The structure of the data dir is as follows:

![image](https://github.com/user-attachments/assets/835800cf-e607-43c3-b2c6-68baddb12259)

MELD(Plz check out the following sharing link)

https://www.dropbox.com/s/edspgpbgnouh21h/MELD_revised.zip?dl=0

https://www.dropbox.com/s/5m6rcg5g2nhys22/MELD.zip?dl=0

In order to implement the proposed SKIER framework, you have to download the pre-trained GloVe vectors(glove.6B.100d.txt is the most commonly used vectors in this project). The downloaded GloVe vectors should be placed in glove dir(plz create glove dir if empty). Note that the batch size should be set to 1 as we process one dialogue each time.

Check out GloVe Embeddings https://nlp.stanford.edu/data/glove.6B.zip before you run the code.

To run this code, plz use the following command (take meld dataset as an example)

python3 train_dd.py --model-type roberta_large --att_dropout 0.5 --output_dim 1024 --chunk_size 50 --base-lr 0.0000005  --epochs 15 --num_epochs 40 --num_relations 10 --data_type meld --num_features 6 --freeze_glove --num_class 7 --use_fixed



