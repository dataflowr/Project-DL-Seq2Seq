from numpy.random import randint
from eval_hand import load_pretrained_uncond, gauss_params_plot, plot_stroke
from model import model_uncond, mdn_loss, sample_uncond, scheduled_sample

lr_model, h_size = load_pretrained_uncond()

strokes, mix_params = sample_uncond(lr_model, h_size, random_state=randint(100))
plot_stroke(strokes)
gauss_params_plot(mix_params)