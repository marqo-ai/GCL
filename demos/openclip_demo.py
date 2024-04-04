import torch
from PIL import Image
import open_clip
import wget

model_url = "https://marqo-gcl-public.s3.us-west-2.amazonaws.com/v1/gcl-vitb32-117-gs-full-states.pt"
wget.download(model_url, "gcl-vitb32-117-gs-full-states.pt")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='gcl-vitb32-117-gs-full-states.pt')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open('assets/oxford_shoe.png')).unsqueeze(0)
text = tokenizer(["a dog", "Vintage Style Women's Oxfords", "a cat"])
logit_scale = 10
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)