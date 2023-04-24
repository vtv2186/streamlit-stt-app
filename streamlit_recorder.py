"""
recorder based on streamlit-webrtc 
streamlit run st_recorder.py --server.port 8606
"""
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    WebRtcStreamerContext,
    ClientSettings
    
)
from aiortc.contrib.media import MediaRecorder
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import queue
from pathlib import Path
import time
import pydub
from streamlit_server_state import server_state, server_state_lock


# from streamlit_lottie import st_lottie
import json

# file_ = '16581-audio.json'
# with open(file_, 'r', encoding='utf-8') as f:
#     lottie_json = json.load(f)

TMP_DIR = Path('temp')
if not TMP_DIR.exists():
    TMP_DIR.mkdir(exist_ok=True, parents=True)

MEDIA_STREAM_CONSTRAINTS = {
    "video": False,
    "audio": {
        # these setting doesn't work
        # "sampleRate": 48000,
        # "sampleSize": 16,
        # "channelCount": 1,
        "echoCancellation": False,  # don't turn on else it would reduce wav quality
        "noiseSuppression": True,
        "autoGainControl": True,
    },
}


def aiortc_audio_recorder(wavpath):
    def recorder_factory():
        return MediaRecorder(wavpath)

    if "webrtc_contexts" not in server_state:
        server_state["webrtc_contexts"] = []   
 
   # webrtc_ctx: WebRtcStreamerContext = webrtc_streamer(
    webrtc_ctx=webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDRECV,
        #mode=WebRtcMode.SENDRECV,
        in_recorder_factory=recorder_factory,
        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        )
        #media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        #client_settings=ClientSettings(
        # rtc_configuration={
        #    # "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        #    "iceServers": [{
        #       "urls": [ "stun:ws-turn4.xirsys.com" ]
        #    }, {
        #       "username": "UIvu1OpNVH8Aw_IWuAYaSU2o6WaTD2hyykLgfqkO563ivxUWWAfnguGDIar3AaoaAAAAAGQrHyp2aXNobnV0ZWph",
        #       "credential": "eebe884a-d24f-11ed-9d96-0242ac140004",
        #       "urls": [
        #           "turn:ws-turn4.xirsys.com:80?transport=udp",
        #           "turn:ws-turn4.xirsys.com:3478?transport=udp",
        #           "turn:ws-turn4.xirsys.com:80?transport=tcp",
        #           "turn:ws-turn4.xirsys.com:3478?transport=tcp",
        #           "turns:ws-turn4.xirsys.com:443?transport=tcp",
        #           "turns:ws-turn4.xirsys.com:5349?transport=tcp"
        #       ]
        #    }]
        #        },
        #        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,

        # ),
        #        )
              
    with server_state_lock["webrtc_contexts"]:
              webrtc_contexts = server_state["webrtc_contexts"]
              if webrtc_ctx.state.playing and webrtc_ctx not in webrtc_contexts:
                  webrtc_contexts.append(webrtc_ctx)
                  server_state["webrtc_contexts"] = webrtc_contexts
              elif not webrtc_ctx.state.playing and webrtc_ctx in webrtc_contexts:
                  webrtc_contexts.remove(webrtc_ctx)
                  server_state["webrtc_contexts"] = webrtc_contexts
                 
                 
    active_other_ctxs = [
              ctx for ctx in webrtc_contexts if ctx != webrtc_ctx and ctx.state.playing
          ]
         
    for ctx in active_other_ctxs:
              webrtc_streamer(
                  key=str(id(ctx)),
                  mode=WebRtcMode.SENDONLY,
                  client_settings=ClientSettings(
                  rtc_configuration={
                    # "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    "iceServers": [{
                        "urls": [ "stun:ws-turn4.xirsys.com" ]
                    }, {
                        "username": "UIvu1OpNVH8Aw_IWuAYaSU2o6WaTD2hyykLgfqkO563ivxUWWAfnguGDIar3AaoaAAAAAGQrHyp2aXNobnV0ZWph",
                        "credential": "eebe884a-d24f-11ed-9d96-0242ac140004",
                        "urls": [
                            "turn:ws-turn4.xirsys.com:80?transport=udp",
                            "turn:ws-turn4.xirsys.com:3478?transport=udp",
                            "turn:ws-turn4.xirsys.com:80?transport=tcp",
                            "turn:ws-turn4.xirsys.com:3478?transport=tcp",
                            "turns:ws-turn4.xirsys.com:443?transport=tcp",
                            "turns:ws-turn4.xirsys.com:5349?transport=tcp"
                        ]
                    }]
                        },
                        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,

                  ),
                        source_audio_track=ctx.output_audio_track,
                        source_video_track=ctx.output_video_track,
                        desired_playing_state=ctx.state.playing,
                        )
   
def save_frames_from_audio_receiver(wavpath):
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY, #convert back SENDONLY if this does not work
        #mode=WebRtcMode.SENDRECV,
       # media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
    )
    
   
           # client_settings=ClientSettings(
            #    rtc_configuration={  # Add this line
          # "iceServers": [{"urls": ["stun:stun4.l.google.com:19302"]}]
        #   "iceServers": [{   "urls": [ "stun:ws-turn1.xirsys.com" ]}, {   "username": "_gOvGuKm6kPUXW7I78axfsf8e7hlY2VaJziOfzYjFnnEZqUb50vvQhQQzevloqKTAAAAAGQc1ox2aXNobnV0ZWph",   "credential": "6b2aa7c4-c9cc-11ed-b509-0242ac140004",   "urls": [       "turn:ws-turn1.xirsys.com:80?transport=udp",       "turn:ws-turn1.xirsys.com:3478?transport=udp",       "turn:ws-turn1.xirsys.com:80?transport=tcp",       "turn:ws-turn1.xirsys.com:3478?transport=tcp",       "turns:ws-turn1.xirsys.com:443?transport=tcp",       "turns:ws-turn1.xirsys.com:5349?transport=tcp"   ]}]
      # "iceServers": [{
      #     "urls": [ "stun:ws-turn4.xirsys.com" ]
      # }, {
      #     "username": "UIvu1OpNVH8Aw_IWuAYaSU2o6WaTD2hyykLgfqkO563ivxUWWAfnguGDIar3AaoaAAAAAGQrHyp2aXNobnV0ZWph",
      #     "credential": "eebe884a-d24f-11ed-9d96-0242ac140004",
      #     "urls": [
      #         "turn:ws-turn4.xirsys.com:80?transport=udp",
      #         "turn:ws-turn4.xirsys.com:3478?transport=udp",
      #         "turn:ws-turn4.xirsys.com:80?transport=tcp",
      #         "turn:ws-turn4.xirsys.com:3478?transport=tcp",
      #         "turns:ws-turn4.xirsys.com:443?transport=tcp",
      #         "turns:ws-turn4.xirsys.com:5349?transport=tcp"
      #     ]
      # }]
      #   },
            
          # source_audio_track=ctx.output_audio_track,
          # source_video_track=ctx.output_video_track,
          #  desired_playing_state=ctx.state.playing,
        

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    status_indicator = st.empty()
    lottie = False
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                status_indicator.info("No frame arrived.")
                continue

            # if not lottie:  # voice gif
            #     st_lottie(lottie_json, height=80)
            #     lottie = True

            for i, audio_frame in enumerate(audio_frames):
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                # st.markdown(f'{len(audio_frame.layout.channels)}, {audio_frame.format.bytes}, {audio_frame.sample_rate}')
                # 2, 2, 48000
                st.session_state["audio_buffer"] += sound
        else:
            lottie = True
            break

    audio_buffer = st.session_state["audio_buffer"]

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        audio_buffer.export(wavpath, format="wav")
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()


def display_wavfile(wavpath):
    audio_bytes = open(wavpath, 'rb').read()
    file_type = Path(wavpath).suffix
    st.audio(audio_bytes, format=f'audio/{file_type}', start_time=0)


def plot_wav(wavpath):
    audio, sr = sf.read(str(wavpath))
    fig = plt.figure()
    plt.plot(audio)
    plt.xticks(
        np.arange(0, audio.shape[0], sr / 2), np.arange(0, audio.shape[0] / sr, 0.5)
    )
    plt.xlabel('time')
    st.pyplot(fig)


def record_page():
    st.markdown('# recorder')
    if "wavpath" not in st.session_state:
        cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        tmp_wavpath = TMP_DIR / f'{cur_time}.wav'
        st.session_state["wavpath"] = str(tmp_wavpath)

    wavpath = st.session_state["wavpath"]

    aiortc_audio_recorder(wavpath)  # first way
    # save_frames_from_audio_receiver(wavpath)  # second way

    if Path(wavpath).exists():
        st.markdown(wavpath)
        display_wavfile(wavpath)
        plot_wav(wavpath)


if __name__ == "__main__":
    record_page()
