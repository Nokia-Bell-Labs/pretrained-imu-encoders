import torch
import clip
import random
import numpy as np
# from imagebind import data
import torch
# from imagebind.models import imagebind_model
# from imagebind.models.imagebind_model import ModalityType
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        if weights is not None:
            output_size, input_size = weights.shape
        else:
            output_size, input_size = 8, 512 # CHANGE THIS!!!!
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)
    

def get_text_embeddings(model_type="ViT-B/16", text_prompts=None, embed_dim=512):
    """
    Compute text embeddings for each prompt
    """
    clip_encoder = CLIPTextModelWithProjection.from_pretrained("/mnt/nfs/projects/usense/data/clip4clip-webvid150k").cuda()
    tokenizer = CLIPTokenizer.from_pretrained("/mnt/nfs/projects/usense/data/clip4clip-webvid150k")

    # compute text embeddings for each prompt
    # clip_encoder, _ = clip.load(model_type, device="cuda")
    with torch.no_grad():
        text_embeddings = []
        for _, texts in text_prompts.items():
            class_tensor = torch.zeros(1, embed_dim).cuda()  # Initialize class tensor for averaging
            for text in texts:
                # text_tokens = clip.tokenize([text]).cuda()
                with torch.no_grad():
                    # text_embedding = clip_encoder.encode_text(text_tokens)
                    inputs = tokenizer(text=text , return_tensors="pt")
                    text_embedding = clip_encoder(input_ids=inputs["input_ids"].cuda(), attention_mask=inputs["attention_mask"].cuda())[0]
                    text_embedding = text_embedding/text_embedding.norm(dim=-1, keepdim=True)

                class_tensor += text_embedding
            class_tensor /= len(texts)  # Average the embeddings for the class
            text_embeddings.append(class_tensor.cpu())

        text_embeddings = torch.cat(text_embeddings, dim=0).cuda()

    return text_embeddings

def get_zeroshot_classifier(model_type="ViT-B/16", text_prompts=None, embed_dim=512, num_classes=8):
    if text_prompts is not None:
        print("Initializing classification head with text embeddings...")
        zeroshot_weights = get_text_embeddings(model_type, text_prompts=text_prompts, embed_dim=embed_dim)
        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    else:
        print("Randomly initializing classification head...")
        classification_head = torch.nn.Linear(embed_dim, num_classes)
    return classification_head

def get_zeroshot_classifier_imagebind(model, text_prompts=None):

    weights = []
    for i,text_list in text_prompts.items():
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text_list, 'cuda')
        }
        with torch.no_grad():
            embeddings = model(inputs)[ModalityType.TEXT]
            embeddings = embeddings/embeddings.norm(dim=-1, keepdim=True)
            # embeddings = [emb/emb.norm(dim=-1, keepdim=True) for emb in embeddings]

        weights.append(torch.mean(embeddings, dim=0))
        

    weights = torch.stack(weights, dim=0)
    classification_head = ClassificationHead(normalize=True, weights=weights)

    return classification_head

def set_random_seed(seed: int) -> None:
	"""
	Sets the seeds at a certain value.
	:param seed: the value to be set
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic=  True