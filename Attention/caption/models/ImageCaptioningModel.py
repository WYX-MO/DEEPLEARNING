# ImageCaptioningModel.py

import torch.nn as nn
import torch
import sys
import os 
sys.path.insert(0,os.path.join( os.path.dirname(os.path.abspath(__file__)),'..','..','..'))
from Attention.common.vision.Vit import VisionTransformer
from Attention.common.text_embedding import TextEmbedding
from Attention.caption.models.Decoder_block import DecoderBlock
##from Attention.models.PositionalEncoding import PositionalEncoding
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size,max_seq_len,patch_size = 14,decoder_nums = 1):
        super().__init__()
        #image vit
        self.max_seq_len = max_seq_len  # Store the maximum sequence length for text
        self.image_vit = VisionTransformer(
        in_channels=3,
        patch_size=patch_size,
        d_model=192,
        num_layers=12,
        num_heads=3,
        mlp_ratio=4.0,
        cls=False)  # Initialize the Vision Transformer for image processing

        self.text_embeding = TextEmbedding(
            vocab_size = vocab_size,
            d_model = 192,
            max_seq_len=max_seq_len)  # Initialize the text embedding module
        self.decoder_block = nn.ModuleList([DecoderBlock(192, 192, 3) for _ in range(decoder_nums)])  # Initialize the decoder block
        self.pred_head = nn.Linear(192, vocab_size)  # Initialize the prediction head to map decoder output to vocabulary size

    def forward(self, images, captions ):
        image_features = self.image_vit(images)  # Get image features from the Vision Transformer
        text_features = self.text_embeding(captions)  # Get text features from the text embedding module
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len,device = images.device))  # Create a mask to prevent attending to future tokens
        decoder_output = text_features
        for decoder_block in self.decoder_block: 
            decoder_output = decoder_block(image_features, decoder_output, mask[:decoder_output.shape[1], :decoder_output.shape[1]])  # Pass image and text features through the decoder block

        decoder_output = self.pred_head(decoder_output)  # Apply the prediction head to the decoder output
        return decoder_output
    
    @staticmethod
    def caption_shifting(captions):
        return captions[:, :-1],captions[:,1:]  # Shift the captions to the right by one position

if __name__ == "__main__":
    vocab_size = 10000
    model = ImageCaptioningModel(vocab_size, 32, patch_size=4)

    images = torch.randn(2, 3, 224, 244)
    captions = torch.randint(0, vocab_size, (2, model.max_seq_len))

    logits = model(images, captions)

    print(logits.shape)