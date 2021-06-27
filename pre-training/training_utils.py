def return_layerwise_decay_optim(model, peak_lr = 5e-4, rho = 2.6, num_layers_lm = 12):
    '''
        peak lr - input side
        rho - layer-wise decay parameter
    '''
    from torch.optim import AdamW
    twelve = ["roberta.encoder.layer.{}.".format(i) for i in range(num_layers_lm)]
    categories = ["roberta.embeddings."]
    categories.extend(twelve)
    categories.append("lm_head.")

    lr_list = []

    for i in range(len(categories)):
        lr_list.append(0)

    current = peak_lr
    for i in range(len(lr_list) - 1, -1, -1):
        lr_list[i] = current
        current = current/rho

    print(categories)
    print(lr_list)
    print(len(categories), len(lr_list))

    optimizer_grouped_parameters = []

    for i in range(len(categories)):
        item = {}
        item['params'] = [p for n, p in model.named_parameters() if categories[i] in n]
        if i < len(categories) - 1: #keep the peak_lr as default for last layer 
            item['lr'] = lr_list[i]
        optimizer_grouped_parameters.append(item)

    print([len(item['params']) for item in optimizer_grouped_parameters])

    optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=peak_lr,
                betas=(0.9, 0.98),
                eps=1e-6,
            )
    return optimizer
