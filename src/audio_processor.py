
from gtts import gTTS


import tempfile

def text_to_mp3(text):
    """
    Convierte un texto a un archivo MP3 temporalmente.
    """
    tts = gTTS(text=text, lang='es', slow=False, tld="us")
    
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        temp_file_path = temp_file.name
        tts.save(temp_file_path)
    
    return temp_file_path