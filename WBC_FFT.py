import logging
import queue

import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st

from streamlit_server_state import server_state, server_state_lock
from streamlit_webrtc import ClientSettings, WebRtcMode, webrtc_streamer


def main():
   # logger = logging.getLogger(__name__)
    if "webrtc_contexts" not in server_state:
        server_state["webrtc_contexts"] = []

    self_ctx = webrtc_streamer(
        key="self",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=256,
        client_settings=ClientSettings(
            rtc_configuration={  # Add this line
      # "iceServers": [{"urls": ["stun:stun4.l.google.com:19302"]}]
      # "iceServers": [{   "urls": [ "stun:ws-turn1.xirsys.com" ]}, {   "username": "_gOvGuKm6kPUXW7I78axfsf8e7hlY2VaJziOfzYjFnnEZqUb50vvQhQQzevloqKTAAAAAGQc1ox2aXNobnV0ZWph",   "credential": "6b2aa7c4-c9cc-11ed-b509-0242ac140004",   "urls": [       "turn:ws-turn1.xirsys.com:80?transport=udp",       "turn:ws-turn1.xirsys.com:3478?transport=udp",       "turn:ws-turn1.xirsys.com:80?transport=tcp",       "turn:ws-turn1.xirsys.com:3478?transport=tcp",       "turns:ws-turn1.xirsys.com:443?transport=tcp",       "turns:ws-turn1.xirsys.com:5349?transport=tcp"   ]}]
   
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
            media_stream_constraints={"video": True, "audio": True},
        ),
        sendback_audio=True,
    )

    # fig_place = st.empty()

    # fig, [ax_time, ax_freq] = plt.subplots(2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2})

    # sound_window_len = 5000  # 5s
    # sound_window_buffer = None
    
    
    with server_state_lock["webrtc_contexts"]:
        webrtc_contexts = server_state["webrtc_contexts"]
        if self_ctx.state.playing and self_ctx not in webrtc_contexts:
            webrtc_contexts.append(self_ctx)
            server_state["webrtc_contexts"] = webrtc_contexts
        elif not self_ctx.state.playing and self_ctx in webrtc_contexts:
            webrtc_contexts.remove(self_ctx)
            server_state["webrtc_contexts"] = webrtc_contexts

    active_other_ctxs = [
        ctx for ctx in webrtc_contexts if ctx != self_ctx and ctx.state.playing
    ]

    for ctx in active_other_ctxs:
        webrtc_streamer(
            key=str(id(ctx)),
            mode=WebRtcMode.SENDRECV,
            client_settings=ClientSettings(
                rtc_configuration={  # Add this line
          # "iceServers": [{"urls": ["stun:stun4.l.google.com:19302"]}]
          # "iceServers": [{   "urls": [ "stun:ws-turn1.xirsys.com" ]}, {   "username": "_gOvGuKm6kPUXW7I78axfsf8e7hlY2VaJziOfzYjFnnEZqUb50vvQhQQzevloqKTAAAAAGQc1ox2aXNobnV0ZWph",   "credential": "6b2aa7c4-c9cc-11ed-b509-0242ac140004",   "urls": [       "turn:ws-turn1.xirsys.com:80?transport=udp",       "turn:ws-turn1.xirsys.com:3478?transport=udp",       "turn:ws-turn1.xirsys.com:80?transport=tcp",       "turn:ws-turn1.xirsys.com:3478?transport=tcp",       "turns:ws-turn1.xirsys.com:443?transport=tcp",       "turns:ws-turn1.xirsys.com:5349?transport=tcp"   ]}]
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
                media_stream_constraints={
                    "video": True,
                    "audio": True,
                },
            ),
            source_audio_track=ctx.output_audio_track,
            source_video_track=ctx.output_video_track,
            desired_playing_state=ctx.state.playing,
        )
    fig_place = st.empty()
    fig, [ax_time, ax_freq] = plt.subplots(2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2})
    sound_window_len = 5000  # 5s
    sound_window_buffer = None
    while True:
        if self_ctx.audio_receiver:
            try:
                audio_frames = self_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                print("audio queue empty")
                #logger.warning("Queue is empty. Abort.")
               # break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]

            if sound_window_buffer:
                # Ref: https://own-search-and-study.xyz/2017/10/27/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%82%92%E4%BD%9C/  # noqa
                sound_window_buffer = sound_window_buffer.set_channels(1)  # Stereo to mono
                sample = np.array(sound_window_buffer.get_array_of_samples())

                ax_time.cla()
                times = (np.arange(-len(sample), 0)) / sound_window_buffer.frame_rate
                ax_time.plot(times, sample)
                ax_time.set_xlabel("Time")
                ax_time.set_ylabel("Magnitude")

                spec = np.fft.fft(sample)
                freq = np.fft.fftfreq(sample.shape[0], 1.0 / sound_chunk.frame_rate)
                freq = freq[: int(freq.shape[0] / 2)]
                spec = spec[: int(spec.shape[0] / 2)]
                spec[0] = spec[0] / 2

                ax_freq.cla()
                ax_freq.plot(freq, np.abs(spec))
                ax_freq.set_xlabel("Frequency")
                ax_freq.set_yscale("log")
                ax_freq.set_ylabel("Magnitude")

                fig_place.pyplot(fig)
        else:
            print("hello world")
           # logger.warning("AudioReciver is not set. Abort.")
           # break

if __name__ == "__main__":
    main()
