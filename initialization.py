from torch.nn.init import xavier_uniform_
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'src_embeddings'):
            model.encoder.src_embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc_src)

        if hasattr(model.encoder, 'mt_embeddings'):
            model.encoder.mt_embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc_mt)

        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    model.generator = generator
    model.to(device)
    if model_opt.model_dtype == 'fp16':
        model.half()