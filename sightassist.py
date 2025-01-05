def generate_audio_alert(label, position):
    """Generate an audio alert file and provide it for download."""
    phrases = [
        f"Be careful, there's a {label} on your {position}.",
        f"Watch out! {label} detected on your {position}.",
        f"Alert! A {label} is on your {position}.",
    ]
    caution_note = random.choice(phrases)

    # Save the audio file locally
    temp_file_path = os.path.join(audio_temp_dir, f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
    tts = gTTS(caution_note)
    tts.save(temp_file_path)
    
    return temp_file_path


# Update the logic where audio alerts are triggered
if audio_activation:
    current_time = datetime.now()
    for label, position in detected_objects.items():
        if label not in last_alert_time or current_time - last_alert_time[label] > alert_cooldown:
            temp_file_path = generate_audio_alert(label, position)
            last_alert_time[label] = current_time

            # Provide the audio file for download
            with open(temp_file_path, "rb") as audio_file:
                st.download_button(
                    label=f"Download alert for {label} on your {position}",
                    data=audio_file,
                    file_name=f"{label}_alert.mp3",
                    mime="audio/mpeg"
                )
