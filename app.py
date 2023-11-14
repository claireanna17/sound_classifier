import streamlit as st
import prediction


def app():
    st.title("Sound Classifier")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with st.form("analysis_form"):
            analyze_button_clicked = st.form_submit_button("Analyze")

        if analyze_button_clicked:
            # Perform analysis on the uploaded file and get the result
            analysis_result = analyze_audio(uploaded_file)

            # Display the analysis result
            st.write("The sounds present are:")
            st.write(analysis_result)

            # Show the reset button
            reset_button_clicked = st.button("Reset")

            if reset_button_clicked:
                # Clear the uploaded file and analysis result
                st.session_state.uploaded_file = None
                st.session_state.analysis_result = None


def analyze_audio(audio_file):
    label = prediction.predict(audio_file)
    return f"Analysis result for {audio_file.name} is '{label}'" # placeholder for now


if __name__ == "__main__":
    app()
