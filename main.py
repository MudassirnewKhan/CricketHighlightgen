import os
import re
import argparse
import logging
import cv2
import numpy as np
import moviepy.editor as mp

from src.audio_processing import get_exact_audio_peaks
from src.video_utils import is_ground_view, is_crowd_view
from src.scoreboard_ocr import find_and_lock_scoreboard, find_component_locations, reader, score_pattern, overs_pattern
from src.clip_splicer import compile_final_moviepy_reel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_full_match_timeline(video_path, overs_loc):
    video = mp.VideoFileClip(video_path)
    ox, oy, ow, oh = overs_loc
    match_map = {}

    def get_over_at_t(t):
        if t >= video.duration: return None
        frame = video.get_frame(t)
        roi = frame[oy:oy+oh, ox:ox+ow]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text_list = reader.readtext(gray, detail=0)
        text = "".join(text_list).replace(' ', '').replace(',', '.')
        match = re.search(r'(\d{1,2})\.', text)
        return int(match.group(1)) if match else None

    logging.info("🛰️ Locating structural match boundaries layout...")
    current_search_start = 0
    for t in range(0, int(video.duration), 5):
        if get_over_at_t(t) == 0:
            current_search_start = t
            logging.info(f"🏏 Match officially identified starting at time node {t}s")
            break

    over_count = 0
    while current_search_start < (video.duration - 10):
        target_val = over_count + 1
        low = current_search_start
        high = min(low + 600, video.duration)

        while (high - low) > 1:
            mid = (low + high) / 2
            obs = get_over_at_t(mid)
            if obs is not None:
                if obs < target_val: low = mid
                else: high = mid
            else:
                low += 1
                if low >= high: break

        if int(high) <= current_search_start + 1:
            match_map[over_count] = [int(current_search_start), int(video.duration)]
            break

        match_map[over_count] = [int(current_search_start), int(high)]
        current_search_start = int(high)
        over_count += 1

    video.close()
    return match_map

def map_all_live_action(video_path, overs_loc, timeline):
    video = mp.VideoFileClip(video_path)
    ox, oy, ow, oh = overs_loc
    live_map = {}

    for over_num, (start, end) in timeline.items():
        live_segments = []
        is_live = False
        segment_start = start

        for t in range(int(start), int(end) + 1):
            frame = video.get_frame(t)
            roi = frame[oy:oy+oh, ox:ox+ow]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray, detail=0)
            has_board = len(results) > 0

            if not is_live and has_board:
                segment_start = t
                is_live = True
            elif is_live and not has_board:
                if t > segment_start:
                    live_segments.append([segment_start, t])
                is_live = False

        if is_live:
            live_segments.append([segment_start, end])
        live_map[over_num] = live_segments

    video.close()
    return live_map

def get_clean_replay_timeline_with_times(live_map, timeline, min_replay_duration=5):
    replay_map = {}
    for over_num, live_segments in sorted(live_map.items()):
        over_start, over_end = timeline[over_num]
        raw_replay_segments = []

        if not live_segments:
            raw_replay_segments.append([over_start, over_end])
        else:
            live_segments = sorted(live_segments, key=lambda x: x[0])
            if live_segments[0][0] > over_start:
                raw_replay_segments.append([over_start, live_segments[0][0]])
            for i in range(len(live_segments) - 1):
                current_end = live_segments[i][1]
                next_start = live_segments[i+1][0]
                if next_start > current_end:
                    raw_replay_segments.append([current_end, next_start])
            if live_segments[-1][1] < over_end:
                raw_replay_segments.append([live_segments[-1][1], over_end])

        clean_replays = [[int(start), int(end)] for start, end in raw_replay_segments if (end - start) >= min_replay_duration]
        replay_map[over_num] = clean_replays
    return replay_map

def extract_clips_full_match_state_math(video_path, live_map, scoreboard_meta, runs_coords, overs_coords, output_folder):
    video = mp.VideoFileClip(video_path)
    rx, ry, rw, rh = runs_coords
    ox, oy, ow, oh = overs_coords

    runs_side = scoreboard_meta['runs_side']
    wickets_side = scoreboard_meta['wickets_side']
    saved, used, tracked_timestamps = [], [], []

    last_stable_over, last_stable_ball, last_w, last_r = None, None, None, None
    runs_at_previous_ball_end, active_ball_tracking_key, has_triggered_this_ball, cooldown = None, None, False, -1

    def overlap(s, e):
        return any(max(s, a) < min(e, b) for a, b in used)

    for over in sorted(live_map.keys()):
        for b_start, b_end in live_map[over]:
            for t in range(int(b_start), int(b_end)):
                is_on_cooldown = (t < cooldown)
                frame = video.get_frame(t)

                text_runs = "".join(reader.readtext(cv2.cvtColor(frame[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2GRAY), detail=0)).replace(" ", "")
                score_m = score_pattern.search(text_runs)

                text_overs = "".join(reader.readtext(cv2.cvtColor(frame[oy:oy+oh, ox:ox+ow], cv2.COLOR_BGR2GRAY), detail=0)).replace(" ", "")
                over_m = overs_pattern.search(text_overs)

                if not score_m or not over_m:
                    continue

                v1, v2 = int(score_m.group(1)), int(score_m.group(2))
                parsed_o, parsed_b = int(over_m.group(1)), (int(over_m.group(2)) if over_m.group(2) else 0)

                if last_stable_over is not None and last_stable_ball is not None:
                    if (parsed_o * 10 + parsed_b < last_stable_over * 10 + last_stable_ball) or (parsed_o > last_stable_over + 1):
                        continue

                o, b = parsed_o, parsed_b
                cur_r = v1 if runs_side == 'left' else v2
                cur_w = v2 if wickets_side == 'right' else v1

                if active_ball_tracking_key is None:
                    runs_at_previous_ball_end = cur_r
                    last_stable_over, last_stable_ball, last_w, last_r = o, b, cur_w, cur_r
                    active_ball_tracking_key = f"{o}.{b}"
                    continue

                current_ball_key = f"{o}.{b}"
                if current_ball_key != active_ball_tracking_key:
                    runs_at_previous_ball_end = last_r if last_r is not None else cur_r
                    active_ball_tracking_key = current_ball_key
                    has_triggered_this_ball = False

                total_runs_change = cur_r - runs_at_previous_ball_end
                wicket_diff = (cur_w - last_w) if last_w is not None else 0

                is_boundary = (not has_triggered_this_ball) and (4 <= total_runs_change <= 8)
                wicket = (wicket_diff == 1)

                if (is_boundary or wicket) and not is_on_cooldown:
                    label = "WICKET" if wicket else "BOUNDARY"
                    start, end = max(b_start, t - 10), min(b_end, t + 4)

                    if not overlap(start, end):
                        clip = video.subclip(start, end)
                        path = os.path.join(output_folder, f"{label}_{o}_{b}_{t}s.mp4")
                        clip.write_videofile(path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                        clip.close()
                        
                        saved.append(path)
                        used.append((start, end))
                        tracked_timestamps.append(t)
                        cooldown = t + 3
                        if is_boundary: has_triggered_this_ball = True

                last_stable_over, last_stable_ball, last_w, last_r = o, b, cur_w, cur_r

    video.close()
    return saved, tracked_timestamps

def extract_near_miss_clips(exact_peaks, tracked_ocr_times, replay_map, video_path, output_folder):
    video = mp.VideoFileClip(video_path)
    
    ocr_windows = [(max(0, ocr_t - 10), ocr_t + 5) for ocr_t in tracked_ocr_times]
    left_out_peaks = [p for p in exact_peaks if not any(w_start <= p <= w_end for w_start, w_end in ocr_windows)]

    if not left_out_peaks:
        video.close()
        return

    blocks, current_block = [], [left_out_peaks[0]]
    for p in left_out_peaks[1:]:
        if p == current_block[-1] + 1: current_block.append(p)
        else:
            blocks.append(current_block)
            current_block = [p]
    blocks.append(current_block)

    valid_audio_events = [b for b in blocks if len(b) >= 3 and b[0] > 60]

    global_replays = []
    for over_num, segments in replay_map.items():
        for r_start, r_end in segments:
            global_replays.append((r_start, r_end))
    global_replays = sorted(global_replays, key=lambda x: x[0])

    near_miss_count = 1
    for event in valid_audio_events:
        roar_start, roar_end = event[0], event[-1]
        
        matched_replay = None
        for r_start, r_end in global_replays:
            if r_start >= (roar_start - 2):
                matched_replay = [r_start, r_end]
                break

        if matched_replay and (0 <= (matched_replay[0] - roar_end) <= 15):
            sample_time = min(matched_replay[0] + 2, matched_replay[1])
            frame_bgr = cv2.cvtColor(video.get_frame(sample_time), cv2.COLOR_RGB2BGR)

            if is_crowd_view(frame_bgr) and not is_ground_view(frame_bgr):
                continue

            clip_start, clip_end = max(0, roar_start - 8), roar_end + 2
            
            path = os.path.join(output_folder, f"NEAR_MISS_SCENE_{near_miss_count}_{clip_start}s.mp4")
            clip = video.subclip(clip_start, clip_end)
            clip.write_videofile(path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            clip.close()
            near_miss_count += 1

    video.close()

def main():
    parser = argparse.ArgumentParser(description="Automated ML Cricket Highlight Generation Pipeline Execution File")
    parser.add_argument("--video", required=True, help="Path to input full match raw broadcast .mp4 video file asset")
    parser.add_argument("--output_dir", default="full_match_highlights", help="Output destination for local clips folder staging")
    parser.add_argument("--final_reel", default="FINAL_MATCH_HIGHLIGHTS.mp4", help="Name of output aggregated target master movie")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    exact_peaks = get_exact_audio_peaks(args.video, sensitivity=1.3)

    scoreboard = find_and_lock_scoreboard(args.video, start_time=600)
    if not scoreboard:
        logging.error("❌ Fatal Error: Could not verify continuous stable scoreboard overlay metrics.")
        return

    x, y, w, h = scoreboard['coords']
    radial_coords = (max(0, x - 50), max(0, y - 50), w + 100, h + 100)

    score_loc, overs_loc, _ = find_component_locations(args.video, wide_coords=radial_coords)

    full_timeline = get_full_match_timeline(args.video, overs_loc)
    live_action_map = map_all_live_action(args.video, overs_loc, full_timeline)
    broadcast_replay_timeline = get_clean_replay_timeline_with_times(live_action_map, full_timeline, min_replay_duration=5)

    _, ocr_action_timestamps = extract_clips_full_match_state_math(
        args.video, live_action_map, scoreboard, score_loc, overs_loc, args.output_dir
    )

    extract_near_miss_clips(exact_peaks, ocr_action_timestamps, broadcast_replay_timeline, args.video, args.output_dir)

    compile_final_moviepy_reel(target_folder=args.output_dir, output_file=args.final_reel, transition_duration=0.4)

if __name__ == "__main__":
    main()