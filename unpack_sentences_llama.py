import requests
import asyncio
import json
from typing import Tuple

class LlamaClient:
    def __init__(self, api_url: str = "http://localhost:11434/api/chat", model: str = "llama2"):
        self.api_url = api_url
        self.model = model

    async def decode_sentence(self, unprocessed_sentence: str) -> Tuple[str, str]:
        # Escape double quotes in the sentence
        escaped_sentence = unprocessed_sentence.replace('"', '\\"')

        # Prepare the messages for the LLaMA model
        messages = [
            {"role": "system", "content": "You are decoding an unprocessed sentence of an English language learner into its original and corrected versions, respectively. The sentence you will be given is tagged with <i></i> tags (errors) and <c></c> tags (corrections). Some sentences may be error free. When the error tag starts with an 'M' as the first letter, the learner missed something in the sentence. Some errors may be missing some word(s) or punctuation; in that case, the tagging convention will only include a <c></c> tag, without a <i></i> tag, but this means that the student didn't originally include what's pointed out as 'missed' in the correction."},
            {"role": "user", "content": "Decode this sentence:'I'd like to complain about your '<NS type=\"RJ\"><NS type=\"RP\"><i>Best</i><c>best</c></NS></NS>' musical show<NS type=\"MP\"><c>,</c></NS> <NS type=\"RA\"><i>that</i><c>which</c></NS> I went to when <NS type=\"UP\"><i>I'd</i><c>I had</c></NS> a week's holiday in London.'"},
            {"role": "assistant", "content": "Original: I'd like to complain about your 'Best' musical show that I went to when I'd a week's holiday in London.\n\nCorrect: I'd like to complain about your ''best' musical show, which I went to when I had a week's holiday in London."},
            {"role": "user", "content": "Decode this: 'Another point which I think was <NS type=\"S\"><i>ennoying,</i><c>annoying,</c></NS> <NS type=\"TV\"><i><NS type=\"AGV\"><i>are</i><c>was</c></NS> the concert halls'"},
            {"role": "assistant", "content": "Original: Another point which I think was annoying, are the concert halls.\n\nCorrect: Another point which I think was annoying was the concert halls."},
            {"role": "user", "content": "Decode this: 'I play basketball a lot and I am a member of our college <NS type=\"RN\"><i>term</i><c>team</c></NS>'"},
            {"role": "assistant", "content": "Original: I play basketball a lot and I am a member of our college term.\n\nCorrect: I play basketball a lot and I am a member of our college team."},
            {"role": "user", "content": f"Decode this: '{escaped_sentence}'"}
        ]

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False  # Disable streaming for simpler response handling
        }

        try:
            # Make request using requests
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # Get the complete response content
            response_text = ""
            try:
                result = response.json()
                if 'message' in result and 'content' in result['message']:
                    response_text = result['message']['content'].strip()
                else:
                    print("Warning: Unexpected response structure:", result)
                    return "Error: Unexpected response structure", "Error: Unexpected response structure"
            except json.JSONDecodeError as e:
                # Handle streaming response if we got one despite requesting non-streaming
                for line in response.text.strip().split('\n'):
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                response_text += chunk['message']['content']
                        except json.JSONDecodeError:
                            continue
            
            # Split the response into original and correct versions
            split_text = response_text.split("\n\n")
            
            if len(split_text) < 2:
                print("Warning: Unexpected response format:", response_text)
                return response_text, response_text
            
            original = split_text[0].replace("Original: ", "").strip()
            correct = split_text[1].replace("Correct: ", "").strip()
            
            return original, correct

        except Exception as e:
            print(f"Error processing response: {e}")
            return str(e), str(e)

async def decode_sentence(unprocessed_sentence: str) -> Tuple[str, str]:
    client = LlamaClient(model="llama2")
    return await client.decode_sentence(unprocessed_sentence)

async def main():
    # Initialize the LLaMA client
    client = LlamaClient(model="llama2")  # or whatever model name you're using
    
    # Test sentence
    unprocessed_sentence = 'We would like to have <NS type="MD"><c>a</c></NS> reply <NS type="M"><c>so we are able</c></NS> to <NS type="RV"><i><NS type="FV"><i>became</i><c>become</c></NS></i><c>begin</c></NS> making the arrangements.'
    
    # Get the decoded output
    decoded_output = await client.decode_sentence(unprocessed_sentence)
    print("Original:", decoded_output[0])
    print("Correct:", decoded_output[1])

if __name__ == "__main__":
    asyncio.run(main())