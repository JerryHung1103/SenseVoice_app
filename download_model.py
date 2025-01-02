from modelscope import snapshot_download

path = "/data/SenseVoice/model"

model_dir = snapshot_download(model_id= 'iic/SenseVoiceSmall', local_dir= path)
print(model_dir)