from model import SentenceVAE
import json
from utils import to_var, idx2word, interpolate
import torch
from nltk.tokenize import TweetTokenizer
from nltk.translate.bleu_score import sentence_bleu
import pandas


def preprocess_sentence(sentence):
    '''
    sentence (str): a sentence
    Return:
        input (tensor): tensor representation using indices
        target (tensor):
        length (int)
    '''

    max_sequence_length=50
    
    words = tokenizer.tokenize(sentence)

    input = ['<sos>'] + words
    input = input[:max_sequence_length]

    target = words[:max_sequence_length-1]
    target = target + ['<eos>']

    assert len(input) == len(target), "%i, %i"%(len(input), len(target))
    length = len(input)

    input.extend(['<pad>'] * (max_sequence_length-length))
    target.extend(['<pad>'] * (max_sequence_length-length))

    input = [w2i.get(w, w2i['<unk>']) for w in input]
    target = [w2i.get(w, w2i['<unk>']) for w in target]

    return input, target, length

def vae_get_sentence(sentence):
    # make a forward pass
    input, target, length = preprocess_sentence(sentence)
    input = torch.tensor([input])
    length = torch.tensor([length])
    logp, mean, logv, z = model(input, length)
    
    # sample a sentence from the latent space
    samples, _ = model.inference(z=z)
    output_str = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])[0]
    output_str = ' '.join(output_str.split()[:-1])
    return output_str

def get_bleu_scores(test, tokenizer, get_sentence):
    '''
    Arguments:
        test (pandas.DataFrame): rows containing the test data
        tokenizer (nltk.tokenize): for tokenizing sentences
        get_sentence (lambda function): take the original sentence as input
            and return an output sentence
    '''

    avg_bleu_orig = 0
    avg_bleu_ref  = 0
    N = 1
    checkpoint = 100

    print('avg_bleu_orig', 'avg_bleu_ref')

    for index, row in test.iterrows():
        
        orig = row['question1']
        ref = row['question2']
        
        orig_words = tokenizer.tokenize(orig)
        ref_words = tokenizer.tokenize(ref)

        output_str = get_sentence(orig)
        output_words = tokenizer.tokenize(output_str)
        
        bleu_orig = sentence_bleu([orig_words], output_words)
        bleu_ref = sentence_bleu([ref_words], output_words)
        
        avg_bleu_orig += (bleu_orig - avg_bleu_orig) / N
        avg_bleu_ref  += (bleu_ref - avg_bleu_ref) / N
        N += 1

        if N % checkpoint == 0:
            print(avg_bleu_orig, avg_bleu_ref)

    return avg_bleu_orig, avg_bleu_ref

if __name__ == '__main__':

    # hard coded for now
    model_path = 'bin/2021-Mar-01-07:50:35/E9.pytorch'
    vocab_f = open('data/quora.vocab.json', 'r')
    vocab = json.load(vocab_f)
    w2i, i2w = vocab['w2i'], vocab['i2w']
    tokenizer = TweetTokenizer(preserve_case=False)

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=50,
        embedding_size=300,
        rnn_type='gru',
        hidden_size=256,
        word_dropout=0,
        embedding_dropout=0.5,
        latent_size=16,
        num_layers=1,
        bidirectional=False
    )

    df = pandas.read_csv('data/quora-question-pairs/train.csv')
    duplicates = df[df['is_duplicate']==1]
    train = duplicates[:50000]
    test = duplicates[-4000:]

    model.load_state_dict(torch.load(model_path))

    # Find the scores for copying sentence
    bleu_orig, bleu_ref = get_bleu_scores(test, tokenizer, lambda x: x)
    print('Scores for copying the original sentence:')
    print('bleu_orig', bleu_orig)
    print('bleu_ref', bleu_ref)

    bleu_orig, bleu_ref = get_bleu_scores(test, tokenizer, vae_get_sentence)
    print('Scores for using the trained VAE:')
    print('bleu_orig', bleu_orig)
    print('bleu_ref', bleu_ref)
