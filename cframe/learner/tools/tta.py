import imgaug.augmenters as iaa
import torch
def tta_only(learn, image, scale:float=1.35):
    "Computes the outputs for several augmented inputs for TTA"

    # augm_tfm = [o for o in learn.data.train_ds.tfms if o.tfm not in
    #            (crop_pad, flip_lr, dihedral, zoom)]

    pbar = range(8)
    for i in pbar:
        print('tta{}'.format(i))
        row = 1 if i&1 else 0
        col = 1 if i&2 else 0
        flip = i&4
        d = {'row_pct':row, 'col_pct':col, 'is_random':False}
        # tfm = [augm_tfm, zoom(scale=scale, **d), crop_pad(**d)]
        tfm = [iaa.CropAndPad(px=(row, row, col, col), keep_size=True)]
        if flip:
            tfm.append(iaa.Fliplr(p=1))

        tmp = image.copy()
        for trans in tfm:
            tmp = trans(images=tmp)
        _, _, pred = learn.predict(tmp)
        yield pred


def TTA(learn, image, beta:float=0.4, scale:float=1.35):
    "Applies TTA to predict on `ds_type` dataset."
    _, _, pred = learn.predict(image)
    all_preds = list(tta_only(learn, image, scale=scale))
    avg_preds = torch.stack(all_preds).mean(0)

    final_preds = pred*beta + avg_preds*(1-beta)
    return final_preds
