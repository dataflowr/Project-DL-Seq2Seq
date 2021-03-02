from eval_nmt import load_pre_trained, evalText, viz_attn

encoder_e2f, decoder_e2f = load_pre_trained('eng-fra')
encoder_f2e, decoder_f2e = load_pre_trained('fra-eng')

eng_text = 'i m not giving you any money .'
fra_text = 'je crains de vous avoir offense .'

inp1, out1, attn1 = evalText(eng_text, encoder_e2f, decoder_e2f)

inp2, out2, attn2 = evalText(fra_text, encoder_f2e, decoder_f2e, inp_lang='French', out_lang='English')

viz_attn(inp1, out1, attn1)
viz_attn(inp2, out2, attn2)