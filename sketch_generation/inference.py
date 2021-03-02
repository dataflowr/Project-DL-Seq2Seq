"""
inference on sketch generation project
"""

import numpy as np
from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss, skrnn_sample
from eval_skrnn import draw_image, load_pretrained_congen, load_pretrained_uncond
import torch

cond_gen = True

if not cond_gen:
    data_type = 'cat' # can be kanji character or cat

    encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_uncond(data_type)
    strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, random_state=30 , \
                                               cond_gen=cond_gen, device=device, bi_mode= mode)
    draw_image(strokes, save=True, save_dir='drawings/unconditional/')

else:
    data_type = 'cat' # can be kanji character or cat
    mode='test' if data_type=='cat' else 'train'

    data_enc, _ , _ = get_data(data_type=data_type, mode=mode) 
    encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type)
    enc_rnd = torch.tensor(data_enc[np.random.randint(0,len(data_enc))], \
                                                                          dtype=torch.float, device =device).unsqueeze(0)

    strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, inp_enc=enc_rnd, \
                                               cond_gen=cond_gen, device=device, bi_mode=mode)
    draw_image(strokes, save=True, save_dir='drawings/conditional/', compare_to=enc_rnd[0, :, [0,1,3]].detach().cpu().numpy())