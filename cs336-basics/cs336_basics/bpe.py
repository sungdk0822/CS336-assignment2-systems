import json
import multiprocessing
import pickle
import regex as re
from collections import Counter
from cs336_basics import config
from cs336_basics.pretokenization_example import find_chunk_boundaries
from pathlib import Path
from typing import Iterable, Iterator, Callable

'''
Problem (train_bpe): BPE Tokenizer Training (15 points)

Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:

input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:

vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
    lary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
'''
def pretokenize(chunk: str, special_tokens: list[str]) -> dict[str, int]:
    if special_tokens:
        escaped_special_tokens = [re.escape(special_token) for special_token in sorted(special_tokens, key=len, reverse=True)]
        pattern = '|'.join(escaped_special_tokens)
        non_capturing_pattern = '(?:' + pattern + ')'
        split_chunks = re.split(non_capturing_pattern, chunk)
    else:
        split_chunks = [chunk]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = {}
    for split_chunk in split_chunks:
        matches = re.finditer(PAT, split_chunk)
        for match in matches:
            pretoken = match.group()
            if pretoken not in pretokens:
                pretokens[pretoken] = 1
            else:
                pretokens[pretoken] += 1
    
    return pretokens


def multiprocess_pretokenize(chunk: str, special_tokens: list[str], output_queue: multiprocessing.Queue):
    output_queue.put(pretokenize(chunk, special_tokens))


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str], 
    num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, 'rb') as f:
        print('bpe train starts...')
        boundaries = find_chunk_boundaries(f, num_processes, '<|endoftext|>'.encode('utf-8'))

        # multiprocessing implementation
        output_queue = multiprocessing.Queue()
        processes = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            process = multiprocessing.Process(target=multiprocess_pretokenize, args=(chunk, special_tokens, output_queue))
            process.start()
            processes.append(process)

        merged_pretokens = Counter()
        for process in processes:
            pretokens = output_queue.get()
            merged_pretokens += Counter(pretokens)

        for process in processes:
            process.join()
        
        merged_pretokens = dict(merged_pretokens)
        byte_pretokens = {tuple([int.to_bytes() for int in pretoken.encode('utf-8')]): count for pretoken, count in merged_pretokens.items()}

        # initialize vocab
        vocab = {token_id: token_id.to_bytes() for token_id in range(256)}
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode('utf-8')
        
        # initialize merges
        merges = []

        while len(vocab) < vocab_size:
            byte_pair_counts = {}
            for token in byte_pretokens:
                for index in range(len(token)-1):
                    byte_pair = (token[index], token[index+1])
                    if byte_pair not in byte_pair_counts:
                        byte_pair_counts[byte_pair] = byte_pretokens[token]
                    else:
                        byte_pair_counts[byte_pair] += byte_pretokens[token]
                        
            max_count_byte_pair = max(sorted(byte_pair_counts, reverse=True), key=lambda byte_pair: byte_pair_counts[byte_pair])

            # merging
            merges.append(max_count_byte_pair)
            vocab[len(vocab)] = max_count_byte_pair[0] + max_count_byte_pair[1]

            # update pretokens by applying the merge
            updated_byte_pretokens = {}
            for token in byte_pretokens:
                updated_token = token
                index = 0
                index_upper_bound = len(updated_token) - 1
                while index < index_upper_bound:
                    byte_pair = (updated_token[index], updated_token[index+1])
                    if byte_pair == max_count_byte_pair:
                        updated_token = updated_token[:index] + (byte_pair[0] + byte_pair[1], ) + updated_token[index+2:]
                        index_upper_bound -= 1
                    index += 1

                updated_byte_pretokens[updated_token] = byte_pretokens[token]

            byte_pretokens = updated_byte_pretokens
            print(f'{len(vocab)} / {vocab_size} ({int(len(vocab) / vocab_size * 100)} percent done)', end='\r')

        print()

    return vocab, merges


def open_results(path):
    path = Path(path)
    if path.exists():
        with open(path, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        return None


def dump_results(path, results):
    path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(results, f)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) 
        a list of special tokens. This function should accept the following parameters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        if '<|endoftext|>' not in self.special_tokens:
            self.special_tokens.append('<|endoftext|>')

        for special_token in self.special_tokens:
            byte_special_token = special_token.encode('utf-8')
            if byte_special_token not in self.vocab.values():
                self.vocab[len(vocab)] = byte_special_token

        self.reversed_vocab = {token: token_id for token_id, token in self.vocab.items()}

    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        '''
        def load_vocab(vocab_filepath: str) -> dict[int, bytes]:
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                reverse_vocab = json.load(f)
            vocab = {token_id: token.encode('utf-8') for token, token_id in reverse_vocab.items()}

            return vocab
        
        def load_merges(merges_filepath: str) -> list[tuple[bytes, bytes]]:
            merges = []
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    pair_left, pair_right = line.split()
                    pair = (pair_left.encode('utf-8'), pair_right.encode('utf-8'))
                    merges.append(pair)

            return merges
            
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str, encode_dictionary: dict[str, tuple[int, ...]] | None = None) -> list[int]:
        '''Encode an input text into a sequence of token IDs.'''
        '''
        Step 1: Pre-tokenize. 
            We first pre-tokenize the sequence and represent each pre-token as a sequence of
            UTF-8 bytes, just as we did in BPE training. We will be merging these bytes within each pre-token into
            vocabulary elements, handling each pre-token independently (no merges across pre-token boundaries).
        Step 2: Apply the merges. 
            We then take the sequence of vocabulary element merges created during BPE
            training, and apply it to our pre-tokens in the same order of creation.
        '''
        if encode_dictionary is None:
            encode_dictionary = {}

        for special_token in self.special_tokens:
            encode_dictionary[special_token] = self.reversed_vocab[special_token.encode('utf-8')]

        def merge_pretoken(pretoken: tuple[bytes, ...]) -> tuple[bytes, ...]:
            merged_token = pretoken
            for merge_pair in self.merges:
                index = 0
                index_upper_bound = len(merged_token) - 1
                while index < index_upper_bound:
                    pair = (merged_token[index], merged_token[index+1])
                    if pair == merge_pair:
                        merged_token = merged_token[:index] + (pair[0] + pair[1], ) + merged_token[index+2:]
                        index_upper_bound -= 1
                    index += 1
            
            return merged_token

        def encode_pretoken(merged_pretoken: tuple[bytes, ...]) -> tuple[int, ...]:
            encoded_pretoken = []
            for token in merged_pretoken:
                encoded_pretoken.append(self.reversed_vocab[token])
            
            return tuple(encoded_pretoken)
        
        escaped_special_tokens = [re.escape(special_token) for special_token in sorted(self.special_tokens, key=len, reverse=True)]
        pattern = '|'.join(escaped_special_tokens)
        capturing_pattern = '(' + pattern + ')'
        split_chunks = re.split(capturing_pattern, text)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        def append_to_encode_dictionary(pretoken: str) -> None:
            byte_pretoken = tuple([int.to_bytes() for int in pretoken.encode('utf-8')])
            merged_pretoken = merge_pretoken(byte_pretoken)
            encoded_pretoken = encode_pretoken(merged_pretoken)
            encode_dictionary[pretoken] = encoded_pretoken

        encoded_text = []
        for split_chunk in split_chunks:
            if split_chunk in self.special_tokens:
                encoded_text.append(encode_dictionary[split_chunk])
                    
            else: # normal chunk
                matches = re.finditer(PAT, split_chunk)
                for match in matches:
                    pretoken = match.group()
                    if pretoken not in encode_dictionary:
                        append_to_encode_dictionary(pretoken)
                    encoded_text.extend(encode_dictionary[pretoken])

        return encoded_text

    def encode_iterable(self, iterable: Iterable[str], use_tqdm: bool = False, max_cache_len: int = 0) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        '''
        from tqdm import tqdm
        encode_dictionary = {}
        iter = tqdm(iterable) if use_tqdm else iterable
        for text in iter:
            if max_cache_len > 0 and len(encode_dictionary) > max_cache_len:
                encode_dictionary = {} # simply clears the cache (a replacement algorithm could be used, but is omitted here)
            yield from self.encode(text, encode_dictionary)

    def decode(self, ids: list[int]) -> str:
        '''Decode a sequence of token IDs into text.'''
        '''
        To decode a sequence of integer token IDs back to raw text, we can simply look up each ID's corresponding
        entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a
        Unicode string.
        Note that input IDs are not guaranteed to map to valid Unicode strings (since a user
        could input any sequence of integer IDs). In the case that the input token IDs do not produce a valid
        Unicode string, you should replace the malformed bytes with the official Unicode replacement character
        U+FFFD. The errors argument of bytes.decode controls how Unicode decoding errors are handled, and
        using errors='replace' will automatically replace malformed data with the replacement marker.
        '''
        concatenated_bytes = b''
        for id in ids:
            concatenated_bytes += self.vocab[id]
        text = concatenated_bytes.decode('utf-8', errors='replace')

        return text


def run_train_bpe(print_vocab = True, find_longest_token = True): 
    vocab_size = 512
    special_tokens = ['<|endoftext|>', '<|example|>']
    
    vocab, merges = train_bpe(config.corpus_path, vocab_size, special_tokens, 1)

    dump_results(config.vocab_path, vocab)
    dump_results(config.merges_path, merges)
    
    if print_vocab:
        vocab = open_results(config.vocab_path)
        print(vocab)

    if find_longest_token and vocab is not None:
        max_token_id = max(vocab, key=lambda token_id: len(vocab[token_id]))
        print(vocab[max_token_id])


def get_profile_func(func: Callable) -> Callable:
    import cProfile
    def profile_func(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        func(*args, **kwargs)

        profiler.disable()
        profiler.print_stats(sort='time')

    return profile_func