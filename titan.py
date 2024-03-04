import boto3
import botocore.config
import json
import base64
from datetime import datetime
from email import message_from_bytes


def extract_text_from_multipart(data):
    msg = message_from_bytes(data)

    text_content = ''

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text_content += part.get_payload(decode=True).decode('utf-8') + "\n"

    else:
        if msg.get_content_type() == "text/plain":
            text_content = msg.get_payload(decode=True).decode('utf-8')

    return text_content.strip() if text_content else None

def read_from_s3_bucket(s3_bucket):
    s3 = boto3.client('s3')
    s3_key = f'summary-input/test-v3.txt'

    try:
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return content.strip() if content else None

    except Exception as e:
        print(f"Error reading file from S3: {e}")


def generate_summary_from_bedrock(content:str, message:str) ->str:
    prompt_data = f"""Answer the question based only on the information provided between ## and give short answers.
    #{content}#
    Question: {message}
    Answer:"""
    
    body = json.dumps({
        "inputText": prompt_data,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0,
            "topP": 1
        }
    })

    try:
        bedrock = boto3.client("bedrock-runtime",region_name="us-west-2",config = botocore.config.Config(read_timeout=300, retries = {'max_attempts':3}))
        response = bedrock.invoke_model(body=body,modelId="amazon.titan-text-express-v1")
        response_content = response.get('body').read().decode('utf-8')
        response_data = json.loads(response_content)
        for result in response_data['results']:
            #print(f"Token count: {result['tokenCount']}")
            #print(f"Output text: {result['outputText']}")
            #print(f"Completion reason: {result['completionReason']}")

            summary = result['outputText'].strip()
        return summary

    except Exception as e:
        print(f"Error generating the summary: {e}")
        return ""

def save_summary_to_s3_bucket(summary, s3_bucket, s3_key):

    s3 = boto3.client('s3')

    try:
        s3.put_object(Bucket = s3_bucket, Key = s3_key, Body = summary)
        print("Summary saved to s3")

    except Exception as e:
        print("Error when saving the summary to s3")


def lambda_handler(event,context):
    s3_bucket = 'bedrock-lambda-api'
    
    #decoded_body = base64.b64decode(event['body'])
    event = json.loads(event['body'])
    message = event['message']

    #text_content = extract_text_from_multipart(decoded_body)
    text_content = read_from_s3_bucket(s3_bucket)

    if not text_content:
        return {
            'statusCode':400,
            'body':json.dumps("Failed to extract content")
        }

    #print(text_content)
    summary = generate_summary_from_bedrock(text_content,message)

    if summary:
        current_time = datetime.now().strftime('%H%M%S') #UTC TIME, NOT NECCESSARILY YOUR TIMEZONE
        s3_key = f'summary-output/{current_time}.txt'
        

        save_summary_to_s3_bucket(summary, s3_bucket, s3_key)

    else:
        print("No summary was generated")


    return {
        'statusCode':200,
        'body':json.dumps(summary)
    }

    
