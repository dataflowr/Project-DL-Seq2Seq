from eval_hand import load_pretrained_congen, gauss_params_plot, plot_stroke, phi_window_plots
from model import model_congen, mdn_loss, sample_congen

lr_model, char_to_vec, h_size = load_pretrained_congen()

# strokes, mix_params, phi, win = sample_congen(lr_model, 'kiki do you love me ?', char_to_vec, h_size)
# plot_stroke(strokes)
# gauss_params_plot(mix_params)
# phi_window_plots(phi, win) 

# strokes, mix_params, phi, win = sample_congen(lr_model, 'a thing of beauty is joy forever', char_to_vec, h_size)
# plot_stroke(strokes)
# gauss_params_plot(mix_params)
# phi_window_plots(phi, win) 

strokes, mix_params, phi, win = sample_congen(lr_model, 'You see, but you do not observe.', char_to_vec, h_size)
plot_stroke(strokes)
# gauss_params_plot(mix_params)
# phi_window_plots(phi, win) 