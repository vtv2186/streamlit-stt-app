import urllib.request
from pathlib import Path

import av
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import queue
import pydub
import logging
from streamlit_server_state import server_state, server_state_lock
from streamlit_webrtc import ClientSettings, WebRtcMode, webrtc_streamer

cv2_path = Path(cv2.__file__).parent


def imread_from_url(url: str):
    req = urllib.request.urlopen(url)
    encoded = np.asarray(bytearray(req.read()), dtype="uint8")
    image_bgra = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    return image_bgra


def overlay_bgra(background: np.ndarray, overlay: np.ndarray, roi):
    roi_x, roi_y, roi_w, roi_h = roi
    roi_aspect_ratio = roi_w / roi_h

    # Calc overlay x, y, w, h that cover the ROI keeping the original aspect ratio
    ov_org_h, ov_org_w = overlay.shape[:2]
    ov_aspect_ratio = ov_org_w / ov_org_h

    if ov_aspect_ratio >= roi_aspect_ratio:
        ov_h = roi_h
        ov_w = int(ov_aspect_ratio * ov_h)
        ov_y = roi_y
        ov_x = int(roi_x - (ov_w - roi_w) / 2)
    else:
        ov_w = roi_w
        ov_h = int(ov_w / ov_aspect_ratio)
        ov_x = roi_x
        ov_y = int(roi_y - (ov_h - roi_h) / 2)

    resized_overlay = cv2.resize(overlay, (ov_w, ov_h))

    # Cut out the pixels of the overlay image outside the background frame.
    margin_x0 = -min(0, ov_x)
    margin_y0 = -min(0, ov_y)
    margin_x1 = max(background.shape[1], ov_x + ov_w) - background.shape[1]
    margin_y1 = max(background.shape[0], ov_y + ov_h) - background.shape[0]

    resized_overlay = resized_overlay[
        margin_y0 : resized_overlay.shape[0] - margin_y1,
        margin_x0 : resized_overlay.shape[1] - margin_x1,
    ]
    ov_x += margin_x0
    ov_w -= margin_x0 + margin_x1
    ov_y += margin_y0
    ov_h -= margin_y0 + margin_y1

    # Overlay
    foreground = resized_overlay[:, :, :3]
    mask = resized_overlay[:, :, 3]

    overlaid_area = background[ov_y : ov_y + ov_h, ov_x : ov_x + ov_w]
    overlaid_area[:] = np.where(mask[:, :, np.newaxis], foreground, overlaid_area)


@st.experimental_singleton
def get_face_classifier():
    return cv2.CascadeClassifier(
        str(cv2_path / "data/haarcascade_frontalface_alt2.xml")
    )


@st.experimental_singleton
def get_filters():
    return {
        "ironman": imread_from_url(
            "https://i.pinimg.com/originals/0c/c0/50/0cc050fd99aad66dc434ce772a0449a9.png"  # noqa: E501
        ),
        "laughing_man": imread_from_url(
            "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/3a17e5a4-9610-4fa3-a4bd-cb7d94d6f7e1/darwcty-d989aaf1-3cfa-4576-b2ac-305209346162.png/v1/fill/w_944,h_847,strp/laughing_man_logo_by_aggressive_vector_darwcty-pre.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9OTE5IiwicGF0aCI6IlwvZlwvM2ExN2U1YTQtOTYxMC00ZmEzLWE0YmQtY2I3ZDk0ZDZmN2UxXC9kYXJ3Y3R5LWQ5ODlhYWYxLTNjZmEtNDU3Ni1iMmFjLTMwNTIwOTM0NjE2Mi5wbmciLCJ3aWR0aCI6Ijw9MTAyNCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.5SDBnNZF6ktZM7Mk5gJfpHNQswRba3eqpvUn6FMHyW4"  # noqa: E501
        ),
        "cat": imread_from_url(
            "https://i.pinimg.com/originals/29/cd/fd/29cdfdf2248ce2465598b2cc9e357579.png"  # noqa: E501
        ),
    }


def main():
    logger = logging.getLogger(__name__)

    if "webrtc_contexts" not in server_state:
        server_state["webrtc_contexts"] = []

    face_cascade = get_face_classifier()
    filters = get_filters()

    filter_type = st.radio(
        "Select filter type",
        ("ironman", "laughing_man", "cat"),
        key="filter-type",
    )
    draw_rect = st.checkbox("Draw rect (for debug)")

 
    
    
    
    
    #fig, [ax_time, ax_freq] = plt.subplots(2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2})

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.11, minNeighbors=3, minSize=(30, 30)
        )

        overlay = filters[filter_type]

        for (x, y, w, h) in faces:
            # Ad-hoc adjustment of the ROI for each filter type
            if filter_type == "ironman":
                roi = (x, y, w, h)
            elif filter_type == "laughing_man":
                roi = (x, y, int(w * 1.15), h)
            elif filter_type == "cat":
                roi = (x, y - int(h * 0.3), w, h)
            overlay_bgra(img, overlay, roi)

            if draw_rect:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    self_ctx = webrtc_streamer(
        key="self",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=256,
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
            media_stream_constraints={"video": True, "audio": True},
        ),
        video_frame_callback=video_frame_callback,
        
       # sendback_audio=False,
    )

    
                
                   
                   
    with server_state_lock["webrtc_contexts"]:
        webrtc_contexts = server_state["webrtc_contexts"]
        if self_ctx.state.playing and self_ctx not in webrtc_contexts:
            webrtc_contexts.append(self_ctx)
            server_state["webrtc_contexts"] = webrtc_contexts
        elif not self_ctx.state.playing and self_ctx in webrtc_contexts:
            webrtc_contexts.remove(self_ctx)
            server_state["webrtc_contexts"] = webrtc_contexts

    fig_place = st.empty()
    fig, [ax_time, ax_freq] = plt.subplots(2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2})
    #ax_time.plot([1,2,3,4])
    #ax_freq.plot([1,2,3,4])
    sound_window_len = 5000  # 5s
    sound_window_buffer = None
     #fig_place.pyplot(fig)

    
    if self_ctx.audio_receiver:
        try:
            audio_frames = self_ctx.audio_receiver.get_frames(timeout=10)
        except queue.Empty:
            logger.warning("Queue is empty. Abort.")
            
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
        logger.warning("AudioReciver is not set. Abort.")
       # break

    

    active_other_ctxs = [
        ctx for ctx in webrtc_contexts if ctx != self_ctx and ctx.state.playing
    ]

    for ctx in active_other_ctxs:
        webrtc_streamer(
            key=str(id(ctx)),
            mode=WebRtcMode.RECVONLY,
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
                media_stream_constraints={
                    "video": True,
                    "audio": True,
                },
            ),
            source_audio_track=ctx.output_audio_track,
            source_video_track=ctx.output_video_track,
            desired_playing_state=ctx.state.playing,
       

        
        )
    
if __name__ == "__main__":
    main()
