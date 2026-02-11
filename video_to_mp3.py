# Converts the videos to mp3

import os
import subprocess

files = os.listdir("videos for rag-ai")

for file in files:
    file_number = file.split(".")[0].split(" #")[1]
    file_name = file.split(" v")[0]
    print(file_number,file_name)
    subprocess.run(["ffmpeg", "-i", f"videos for rag-ai/{file}", f"audios/{file_number}_{file_name}.mp3"])