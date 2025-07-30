import os

TinyStories_train_set_path = 'cs336_basics/data/TinyStoriesV2-GPT4-train.txt'
TinyStories_validation_set_path = 'cs336_basics/data/TinyStoriesV2-GPT4-valid.txt'
OpenWebText_train_set_path = 'cs336_basics/data/owt_train.txt'
OpenWebText_validation_set_path = 'cs336_basics/data/owt_valid.txt'

bpe_train_corpus_path = TinyStories_validation_set_path
corpus_path = TinyStories_train_set_path
validation_corpus_path = TinyStories_validation_set_path
# bpe_train_corpus_path = OpenWebText_validation_set_path
# corpus_path = OpenWebText_train_set_path
# validation_corpus_path = OpenWebText_validation_set_path

corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
vocab_path = f'cs336_basics/vocab-{corpus_name}.pickle'
merges_path = f'cs336_basics/merges-{corpus_name}.pickle'

tokenizer_path = 'cs336_basics/checkpoint/tokenizer.pth'
corpus_ids_path = 'cs336_basics/checkpoint/corpus_ids'
validation_corpus_ids_path = 'cs336_basics/checkpoint/validation_corpus_ids'
checkpoint_dir = 'cs336_basics/checkpoint/' # must end with '/'