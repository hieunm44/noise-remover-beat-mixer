import streamlit as st
import librosa
import torch
import torchaudio
from PIL import Image
from denoise_model import DenoiseModel
from scipy.io.wavfile import write as wavwrite


def denoise(audiofile, denoise_model):
    noisy_audio, org_sampling_rate = torchaudio.load(f'original_audio/{audiofile.name}')
    noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True) if noisy_audio.shape[0]!=1 else noisy_audio # Convert to mono
    noisy_audio = torchaudio.functional.resample(noisy_audio,
                                                 orig_freq=org_sampling_rate,
                                                 new_freq=denoise_model.config['trainset_config']['sample_rate'])
    noisy_audio = noisy_audio.to(device)
    noisy_audio = noisy_audio.unsqueeze(0)
    with torch.no_grad():
        generated_audio = denoise_model.forward(noisy_audio)
        for i in range(2):
            generated_audio = denoise_model.forward(generated_audio)

    return generated_audio


def add_beat(y_h,z_h, weight=0.5):
    if y_h.shape[0] < z_h.shape[0]:
        out= weight*y_h + (1-weight)* z_h[:y_h.shape[0]] 
    else:
        out= weight*y_h[:z_h.shape[0]] + (1-weight)* z_h
    return out


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    denoise_model = DenoiseModel(config_path='configs/DNS-large-full.json',
                                 model_path='exp/DNS-large-full/checkpoint/pretrained.pkl',
                                 device='cuda')
    denoise_model.eval()
    denoise_model.to(device)
    image = Image.open('audio_mixer.jpg')
    st.image(image, width=100)
    st.title('Noise Remover and Beat Mixer')

    # audio file
    voice_file = st.file_uploader('Choose a vocal file', type= ['.wav','.wave', '.mp3'])
    if voice_file is not None:
        # To read file as bytes:
        voice_bytes = voice_file.read()
        with open(f'original_audio/{voice_file.name}', mode='wb') as f:
            f.write(voice_file.getvalue())
        st.audio(voice_bytes, format='audio/wav')
        voice_basename = voice_file.name.split('.')[0]
        denoised_filepath=f'denoised_audio/{voice_basename}_denoised.wav'

    if st.button('Denoise'):
        st.write('')
        denoised = denoise(voice_file, denoise_model)
        out = wavwrite(filename=denoised_filepath,
                       rate=16000,
                       data=denoised.cpu().numpy())
        audio = open(denoised_filepath, 'rb')
        byte = audio.read()
        st.write('Denoised')
        st.audio(byte, format ='audio/wav')
    
    # beat file
    beat_file = st.file_uploader('Choose a beat file', type= ['.wav','.wave', '.mp3'])
    if beat_file is not None:
        beat_bytes = beat_file.read()
        with open(f'beat/{beat_file.name}', mode='wb') as f:
            f.write(beat_file.getvalue())
        st.audio(beat_bytes, format='audio/wav')

    if st.button('Combine beat'):
        
        y, sr = librosa.load(denoised_filepath, mono =True)
        y_hat= librosa.resample(y, orig_sr=sr, target_sr=16000)

        #vocal denoised
        z,srz=  librosa.load(f'beat/{beat_file.name}', mono =True)
        z_hat= librosa.resample(z, orig_sr=sr, target_sr=16000)
 
        out = add_beat(y_hat, z_hat, weight=0.6)

        beat_basename = beat_file.name.split('.')[0]
        combined_filepath = f'combined_audio/{voice_basename}_{beat_basename}_combined.wav'
        wavwrite(filename=combined_filepath, 
                 rate=16000,
                 data=out)
        merged=open(combined_filepath, 'rb')
        byte_merged = merged.read()
        st.write('Combined')
        st.audio(byte_merged, format ='audio/wav')

    