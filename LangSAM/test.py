from PIL import Image
from lang_sam import LangSAM
import cv2
model = LangSAM()
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "wheel."
results = model.predict([image_pil], [text_prompt])