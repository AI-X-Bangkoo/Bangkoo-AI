from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

class GeminiClient:
    def __init__(self):
        self.client = genai.Client(
            api_key="AIzaSyCA3h_wDN3DEDixRPKTpSteOSLGbFlYri4"  # 보안상 나중에 .env로
        )
        self.model = "gemini-2.0-flash-exp-image-generation"

    def generate_image(self, prompt: str, *images: Image.Image) -> Image.Image:
        contents = [prompt] + list(images)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return Image.open(BytesIO(part.inline_data.data))
        return None
