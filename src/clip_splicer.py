import os
import re
import logging
import moviepy.editor as mp

def compile_final_moviepy_reel(target_folder="full_match_highlights", output_file="FINAL_MATCH_HIGHLIGHTS.mp4", transition_duration=0.4):
    """
    Scans the output highlights staging directory, handles chronological matrix sorting 
    via filename timestamp extraction, applies crossfades, and compiles the final video.
    """
    master_timeline_list = []

    if not os.path.exists(target_folder):
        logging.error(f"❌ Target clips directory '{target_folder}' does not exist.")
        return

    # Scan workspace files
    for file_name in os.listdir(target_folder):
        if file_name.endswith('.mp4') and not file_name.startswith('FINAL_') and not file_name.startswith('final_'):
            match = re.search(r'_(\d+)s\.mp4', file_name)
            if match:
                timestamp = int(match.group(1))
                event_type = "Near-Miss" if "NEAR_MISS" in file_name else ("Wicket" if "WICKET" in file_name else "Boundary")
                
                master_timeline_list.append({
                    'file_path': os.path.join(target_folder, file_name),
                    'timestamp': timestamp,
                    'event_type': event_type
                })

    if not master_timeline_list:
        logging.warning("⚠️ No valid individual highlight clip components found to combine.")
        return

    # Master chronological sorting line
    master_timeline_list = sorted(master_timeline_list, key=lambda x: x['timestamp'])

    print("=" * 95)
    print(f"🎬  MOVIEPY CHRONOLOGICAL CONCATENATION REEL (FOLDER: '{target_folder}')")
    print("=" * 95)
    print(f"{'Order':<6} | {'Event Type':<12} | {'Timestamp':<10} | {'File Name'}")
    print("-" * 95)
    for idx, clip_info in enumerate(master_timeline_list, 1):
        print(f"[{idx:02d}]   | {clip_info['event_type']:<12} | {clip_info['timestamp']:>5}s     | {os.path.basename(clip_info['file_path'])}")
    print("=" * 95)

    logging.info("🔄 Initializing binary video stream concatenation...")
    video_objects = []

    try:
        for clip_info in master_timeline_list:
            video_obj = mp.VideoFileClip(clip_info['file_path'])
            if transition_duration > 0:
                video_obj = video_obj.fadein(transition_duration).fadeout(transition_duration)
            video_objects.append(video_obj)

        logging.info("🔗 Stitching layout components using specialized structural composition method...")
        final_reel = mp.concatenate_videoclips(video_objects, method="compose")

        logging.info(f"💾 Exporting aggregated compilation matrix directly to: '{output_file}'")
        final_reel.write_videofile(
            output_file,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            remove_temp=True,
            verbose=False,
            logger='bar'  # Production layout rendering status bar
        )

        final_reel.close()
        for obj in video_objects:
            obj.close()
        logging.info("🏆 Master compilation cycle executed successfully.")

    except Exception as e:
        logging.error(f"❌ Video compilation script block exception: {str(e)}")