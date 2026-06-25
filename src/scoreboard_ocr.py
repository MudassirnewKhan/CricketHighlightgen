import re
import cv2
import easyocr
import logging
import moviepy.editor as mp

# Initialize the shared OCR Engine globally within the module
reader = easyocr.Reader(['en'], gpu=True)

score_pattern = re.compile(r'(\d+)\s*([/-])\s*(\d+)')
overs_pattern = re.compile(r'(\d+)\s*(?:[.\s\D]*\s*(\d+))?')

def find_and_lock_scoreboard(video_path, start_time=600):
    """Scans the timeline to establish a dynamic orientation lock on the scoreboard graphics ribbon."""
    logging.info(f"🎬 Opening video stream. Running fast scan searching from {start_time}s...")
    video = mp.VideoFileClip(video_path)
    duration = int(video.duration)
    max_search_time = min(start_time + 900, duration)

    for sec in range(start_time, max_search_time, 10):
        frame = video.get_frame(sec)
        results = reader.readtext(frame)

        for (bbox, text, prob) in results:
            clean = text.replace(" ", "")
            match = score_pattern.search(clean)

            if not match:
                continue

            try:
                v1, v2 = int(match.group(1)), int(match.group(3))
                delimiter = match.group(2)
            except ValueError:
                continue

            if len(match.group(1)) < 2 and len(match.group(3)) < 2:
                continue

            # Determine layout side alignment maps
            if v1 > v2:
                runs, wickets = v1, v2
                runs_side, wickets_side = 'left', 'right'
            else:
                runs, wickets = v2, v1
                runs_side, wickets_side = 'right', 'left'

            if not (10 <= runs <= 500 and 0 <= wickets <= 10):
                continue

            # Extract spatial crop metrics
            (tl, tr, br, bl) = bbox
            x, y = int(tl[0]), int(tl[1])
            w, h = int(br[0] - tl[0]), int(br[1] - tl[1])

            # Look-ahead stability verification pass
            verify_time = min(sec + 10, duration - 1)
            frame2 = video.get_frame(verify_time)
            roi = frame2[y:y+h, x:x+w]
            if roi.size == 0: continue

            res2 = reader.readtext(roi, detail=0)
            new_clean = "".join(res2).replace(" ", "")
            verify_match = score_pattern.search(new_clean)

            if verify_match:
                logging.info(f"🎯 LOCK CONFIRMED: Sub-pixel graphic coordinates secured at: X={x}, Y={y}")
                video.close()
                return {
                    'coords': (x, y, w, h),
                    'runs_side': runs_side,
                    'wickets_side': wickets_side,
                    'delimiter': delimiter,
                    'initial_runs': runs,
                    'initial_wickets': wickets,
                    'confirmed_time': sec
                }

    video.close()
    return None

def find_component_locations(video_path, wide_coords, timestamp=900):
    """Splits the score ribbon into dedicated component tracking bounding boxes."""
    video = mp.VideoFileClip(video_path)
    frame = video.get_frame(timestamp)
    rx, ry, rw, rh = wide_coords

    full_ribbon = frame[ry:ry+rh, rx:rx+rw]
    results = reader.readtext(full_ribbon, detail=1)

    score_coords, overs_coords = None, None
    extracted_data = {"runs": None, "wickets": None, "overs": None}
    candidates = []

    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tx, ty, tw, th = int(tl[0]), int(tl[1]), int(tr[0]-tl[0]), int(bl[1]-tl[1])
        gx, gy = rx + tx, ry + ty

        score_match = re.search(r'(\d+)[/-](\d+)', text)
        if score_match:
            v1, v2 = int(score_match.group(1)), int(score_match.group(2))
            if v1 <= 10 or v2 <= 10:
                candidates.append({'type': 'score', 'coords': (gx, gy, tw, th), 'v1': v1, 'v2': v2, 'x': gx})

        over_match = re.search(r'(\d{1,2}\.\d)', text)
        if over_match:
            candidates.append({'type': 'overs', 'coords': (gx, gy, tw, th), 'val': over_match.group(1), 'x': gx})

    score_cands = sorted([c for c in candidates if c['type'] == 'score'], key=lambda x: x['x'])
    over_cands = sorted([c for c in candidates if c['type'] == 'overs'], key=lambda x: x['x'])

    if score_cands:
        best_s = score_cands[0]
        score_coords = best_s['coords']
        if best_s['v1'] <= 10 and best_s['v2'] > 10:
            extracted_data["wickets"], extracted_data["runs"] = best_s['v1'], best_s['v2']
        elif best_s['v2'] <= 10 and best_s['v1'] > 10:
            extracted_data["runs"], extracted_data["wickets"] = best_s['v1'], best_s['v2']
        else:
            extracted_data["runs"], extracted_data["wickets"] = best_s['v1'], best_s['v2']

    if over_cands:
        overs_coords = over_cands[0]['coords']
        extracted_data["overs"] = over_cands[0]['val']

    video.close()
    return score_coords, overs_coords, extracted_data