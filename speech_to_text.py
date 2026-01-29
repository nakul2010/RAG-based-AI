import whisper
import json

model = whisper.load_model("turbo")  # Use tiny, base, small, medium, large model for non-english translation for better results

result = model.transcribe(audio="audios/sample.mp3", language="en", task="transcribe", word_timestamps=False)

# print(result["segments"])

chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f)