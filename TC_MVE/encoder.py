"""Text-Conditioned Medical Visual Encoder: a parameter-free calling interface.
The owning Stage1 model registers the modules. This interface adds no state keys.
"""
class TC_MVE:
    def __init__(self, text_encoder, image_encoder):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
    def encode(self, images, questions):
        text_feats, text_mask = self.text_encoder(questions)
        return self.image_encoder(images, text_feats=text_feats.to(images.device),
                                  text_mask=text_mask.to(images.device), return_patch_tokens=True)
