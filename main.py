import os
import json
import functions_framework

import google.cloud.logging

import vertexai
from vertexai.language_models import TextGenerationModel

PROJECT_ID  = os.environ.get('GCP_PROJECT','-')
LOCATION = os.environ.get('GCP_REGION','-')

client = google.cloud.logging.Client(project=PROJECT_ID)
client.setup_logging()

log_name = "predictText-cloudfunction-log"
logger = client.logger(log_name)

@functions_framework.http
def webhook(request):

    request_json = request.get_json(silent=True)

    if request_json and 'prompt' in request_json:
        prompt = request_json['prompt']
        logger.log(f"Received request for prompt: {prompt}")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = TextGenerationModel.from_pretrained("text-bison@001")
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40
        }

        input_text = request_json['input']
        response = model.predict(
            f"""คุณชื่อน้องเหมียว เป็นพนักงานข่ายหรือ Telesale เวป Ole777.com 
        คุณต้องพยายามให้ลูกค้าสมัตรสมาชิกให้ได้ โดยที่คุณต้องหว่านล้อมให้ลูกค้าสมัตร ที่เวปไซท์ https://www.ole98.com/  แล้วจะได้ไปรับ promotion 200% กับพนักงานอีกท่าน โดยที่หลักจากสมัตรพนักงานท่านนั้นจะติดต่อลูกค้าไปเอง

        input: {input_text}
        output:
        """,
            parameters
        )



       # prompt_response = model.predict(prompt,**parameters)
       # logger.log(f"PaLM Text Bison Model response: {prompt_response.text}")
        logger.log(f"PaLM Text Bison Model response: {response.text}")

    else:
        response = 'No prompt provided.'

    return json.dumps({"response_text":response.text})