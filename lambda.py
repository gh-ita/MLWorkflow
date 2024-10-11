import json
import boto3
import base64
import sagemaker
from sagemaker.serializers import IdentitySerializer

s3 = boto3.client('s3')
ENDPOINT = "image-classification-2024-10-03-15-45-30-028"
THRESHOLD = .9

def lambda_handler_one(event, context):
    """A function to serialize target data from S3"""
    #retrieve the image key and bucket from the even object 
    key = event.get("s3_key")
    bucket = event.get("s3_bucket")
    #save the image in the tmp foler
    s3.dowload_file(bucket, key,"/tmp/image.png")
    #base64 encoding the image to send it in a json object to the step function
    with open("/tmp/image.png", "rb") as f:
        image = base64.b64encode(f.read())
        
    print("Event", event.keys())
    return {
        'statusCode' : 200, 
        'body':
        {
            "image_data":image, 
            "s3_bucket": bucket,
            "s3_key":key, 
            "inferences" : []
        }
    }


def lambda_handler_two(event, context):

    # Decode the image data
    image = base64.b64decode(event.get("image_data"))

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(
        endpoint_name = ENDPOINT,
        sagemaker_session = sagemaker.Session()
    )

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

def lambda_handler_three(event, context):
    
    # Grab the inferences from the event
    inferences = event.get("inferences")
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(i >= THRESHOLD for i in inferences)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }