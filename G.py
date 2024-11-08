import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import gradio as gr
import numpy as np
from PIL import Image
import logging



# 定义模型
class SmallVGG(nn.Module):
    def __init__(self):
        super(SmallVGG, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 加载模型
def load_model():
    try:
        model = SmallVGG()
        model.load_state_dict(torch.load("./small_vgg_svhn.pth",
                                         map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        model.eval()
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


# 图像预处理
def preprocess_image(image):
    try:
        if isinstance(image, dict) and "composite" in image:
            image = np.array(image["composite"], dtype=np.uint8)
        else:
            raise ValueError("Expected 'composite' key in dictionary but got none.")

        if image.ndim == 2:
            image = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 4:
            image = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 3:
            image = Image.fromarray(image)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # 调整大小并标准化
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970))
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return None


# 预测函数
def predict(image):
    model = load_model()
    if model is None:
        return "Error: Model could not be loaded."

    image_tensor = preprocess_image(image)
    if image_tensor is None:
        return "Error in image preprocessing."

    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
            logging.info(f"Predicted probabilities: {probabilities}")
        return {str(i): float(prob) for i, prob in enumerate(probabilities)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Error during prediction."


# Gradio 界面
interface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(crop_size=(32, 32), image_mode='L', type='numpy', brush=gr.Brush()),
    outputs=gr.Label(num_top_classes=10),
    live=True,
    description="Draw a digit (0-9) and the model will classify it",
)

# 启动 Gradio 界面
if __name__ == "__main__":
    interface.launch(share=True)
