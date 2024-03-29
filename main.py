from typing import Iterable, List
from timeit import default_timer as timer

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score

import myNN
import utils

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE:0, TGT_LANGUAGE:1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

"""During training, we need a subsequent word mask that will prevent the model from looking into the future words when making predictions.
We will also need masks to hide source and target padding tokens.
 Below, let's define a function that will take care of both."""
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz)) ==1).transpose(0, 1))
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0,1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0,1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def sequential_transform(*transforms):
    """Helper functiom to club together sequential operations"""
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    """Function to add BOS/EOS and create tensor for input sequence indices"""
    return torch.cat((torch.tensor([BOS_IDX]),
                     torch.tensor(token_ids),
                     torch.tensor([EOS_IDX])))

def collate_fn(batch):
    """Function to collate data samples into batch tensors"""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE,TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # Removing the last token from tgt seq as during training, the model should predict
        #the next token given the previous tokens
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses/len(list(train_dataloader))

def evaluate(model):
    model.eval()
    losses = 0 

    val_iter = Multi30k(root='D:\\IT\\AI\\Pytorch_Tutorial\\data',
                        split='valid', language_pair=(SRC_LANGUAGE,TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1,:]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_output = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        losses += loss.item()
    return losses/len(list(val_dataloader))

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """Function to generate output sequence using greedy algorithm"""
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:,-1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1,1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str,
              src_lang: str, tgt_lang: str):
    model.eval()
    src = text_transform[src_lang](src_sentence).view(-1,1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens+5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[tgt_lang].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>","").replace("<eos>","")

def evaluate_with_Bleu(data, model, src_lang, tgt_lang):
    targets, outputs = [], []
    count = 0
    for src_sentence, tgt_sentence in data:
        tgt_tokens = token_transform[tgt_lang](tgt_sentence)
        targets.append([tgt_tokens])
        prediction = translate(model, src_sentence, src_lang, tgt_lang)
        output_tokens = token_transform[tgt_lang](prediction)
        outputs.append(output_tokens)
        if count < 5:
            print(f"output_tokens: {output_tokens}\n tgt_tokens: {tgt_tokens}")
        count += 1
    return bleu_score(outputs, targets)



if __name__ == "__main__":
    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

    SRC_LANGUAGE = 'de'
    TGT_LANGUAGE = 'en'

    # Place-holders
    token_transform = {}
    vocab_transform = {}

    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data iterator
        train_iter = Multi30k(root='D:\\IT\\AI\\Pytorch_Tutorial\\data',
                              split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
    
    # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
    # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the parameters of our model
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    LOAD_MODEL = True
    VALID_MODE = True
    
    model = myNN.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                          EMB_SIZE, NHEAD, SRC_VOCAB_SIZE,
                                          TGT_VOCAB_SIZE, FFN_HID_DIM)
    
    

    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.98), eps=1e-9)

    # Load model
    if LOAD_MODEL:
        model, optimizer = utils.load_checkpoint('checkpoint\\checkpoint_at_epoch_1.pth.tar', model, optimizer)

    # Collation
    """Convert string pairs into batched tensors that can be processed by Seq2Seq"""
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transform(token_transform[ln], #tokenization
                                                  vocab_transform[ln],  #Numericalization
                                                  tensor_transform)      # Add BOS/EOS and create tensor
    
    # Training
    NUM_EPOCHS = 1
    SAVE_MODEL = True

    print(translate(model, "Eine Gruppe von Menschen steht vor einem Iglu .", SRC_LANGUAGE, TGT_LANGUAGE))
    if VALID_MODE:
        validation_iter = Multi30k(root='D:\\IT\\AI\\Pytorch_Tutorial\\data',
                                split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        validation_score = evaluate_with_Bleu(validation_iter, model, SRC_LANGUAGE, TGT_LANGUAGE)
        print(f"Bleu score on validation data: {validation_score}")
    else:
        for epoch in range(1, NUM_EPOCHS+1):
            start_time = timer()
            train_loss= train_epoch(model, optimizer)
            end_time = timer()
            val_loss = evaluate(model)
            print(f"Epoch: {epoch}, \nTrain loss: {train_loss:.3f}, \nVal loss: {val_loss:.3f}\n")
            print(f"Epoch time: {(end_time-start_time):.3f}s")
            
            if SAVE_MODEL:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(checkpoint, filename=f'checkpoint\\checkpoint_at_epoch_{epoch}.pth.tar')