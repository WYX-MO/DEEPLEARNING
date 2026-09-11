# image-caption task
This experiment is designed to finish a image caption task,based on flickr8K
dataset.

the model structure is :
- vit : get the image feature
- Attention : handle the captions
- crossAttention decoder: mix the vit feature and captions

## Question
- the vit need a lot of datas and very deep network to get a nice performence,
so i tried to use resnet50 to get image feature instead of train vit from zero by myself.

- the result is bleu4 is ~16 which used resnet50
and ~9 which used tiny-vit

