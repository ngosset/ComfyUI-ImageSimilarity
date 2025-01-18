import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF

def prepare_image_for_resnet(image):
    transform = T.Compose([
        T.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (B, H, W, C) to (B, C, H, W)
        T.Resize(256),
        T.CenterCrop(224),
        T.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),  # Ensure values are 0-1
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )  # ImageNet normalization
    ])

    return transform(image)


class ImageSimilarity:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "image_1": ("IMAGE",),
                    "image_2": ("IMAGE",),
                    "resnet_model": (["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"], {"default": "resnet50"}),
                    "threshold": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0}),
                    }
        }
    
    RETURN_TYPES = ("BOOL", "FLOAT",)
    RETURN_NAMES = ("is_similiar", "cosine_similarity",)
    FUNCTION = "compare_image"
    OUTPUT_NODE = True
    CATEGORY = "Image Similarity"

    def compare_image(self, image_1, image_2, resnet_model, threshold):


        if resnet_model == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif resnet_model == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif resnet_model == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif resnet_model == "resnet101":
            from torchvision.models import resnet101, ResNet101_Weights
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif resnet_model == "resnet152":
            from torchvision.models import resnet152, ResNet152_Weights
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
        
        model.to(self.device)
        model.eval() 

        # Remove the final classification layer to get embeddings
        model = torch.nn.Sequential(*list(model.children())[:-1])
        
        image_1_resized = prepare_image_for_resnet(image_1)
        image_2_resized = prepare_image_for_resnet(image_2)
        
        with torch.no_grad():
            embedding_1 = model(image_1_resized.to(self.device)).squeeze().flatten()
            embedding_2 = model(image_2_resized.to(self.device)).squeeze().flatten()
            
            cos_val = torch.nn.functional.cosine_similarity(embedding_1.unsqueeze(0),
                                                        embedding_2.unsqueeze(0),
                                                        dim=1).item()
        

        return (cos_val >= threshold, cos_val,)
