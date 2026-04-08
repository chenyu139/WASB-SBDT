import csv
import json
import logging
import math
import os.path as osp
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

from dataloaders import build_img_transforms, get_transform
from detectors import build_detector
from trackers import build_tracker
from utils import Center, draw_frame, mkdir_if_missing

from .base import BaseRunner

log = logging.getLogger(__name__)


def _parse_source(source):
    if isinstance(source, int):
        return source
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def _sanitize_score(score):
    score = float(score)
    if math.isfinite(score):
        return score
    return None


class TrajectoryPostprocessor:
    def __init__(self, smooth_alpha, interpolate_max_gap, interpolate_max_disp, interpolate_min_score, max_jump_disp,
                 max_jump_score_threshold, adaptive_jump_scale, adaptive_jump_bias, prediction_min_speed,
                 prediction_error_base, prediction_error_scale, motion_score_threshold):
        self._smooth_alpha = float(smooth_alpha)
        self._interpolate_max_gap = int(interpolate_max_gap)
        self._interpolate_max_disp = float(interpolate_max_disp)
        self._interpolate_min_score = float(interpolate_min_score)
        self._max_jump_disp = float(max_jump_disp)
        self._max_jump_score_threshold = float(max_jump_score_threshold)
        self._adaptive_jump_scale = float(adaptive_jump_scale)
        self._adaptive_jump_bias = float(adaptive_jump_bias)
        self._prediction_min_speed = float(prediction_min_speed)
        self._prediction_error_base = float(prediction_error_base)
        self._prediction_error_scale = float(prediction_error_scale)
        self._motion_score_threshold = float(motion_score_threshold)
        self._pending = []
        self._last_smoothed_xy = None
        self._last_velocity_xy = None

    def push(self, row, frame):
        row_ = dict(row)
        row_['_filled'] = False
        self._pending.append({'row': row_, 'frame': frame, 'smoothed_xy': None})
        self._apply_interpolation()
        self._apply_smoothing()
        return self._flush(final=False)

    def finish(self):
        self._apply_smoothing()
        return self._flush(final=True)

    def _apply_interpolation(self):
        if self._interpolate_max_gap <= 0 or len(self._pending) < 3:
            return

        end_index = len(self._pending) - 1
        end_row = self._pending[end_index]['row']
        if not end_row['visible']:
            return

        start_index = end_index - 1
        while start_index >= 0 and not self._pending[start_index]['row']['visible']:
            start_index -= 1

        gap_size = end_index - start_index - 1
        if start_index < 0 or gap_size <= 0 or gap_size > self._interpolate_max_gap:
            return

        start_row = self._pending[start_index]['row']
        if start_row['x'] is None or start_row['y'] is None or end_row['x'] is None or end_row['y'] is None:
            return
        if start_row['score'] is None or end_row['score'] is None:
            return
        if start_row['score'] < self._interpolate_min_score or end_row['score'] < self._interpolate_min_score:
            return

        dist = math.hypot(end_row['x'] - start_row['x'], end_row['y'] - start_row['y'])
        if self._interpolate_max_disp > 0 and dist > self._interpolate_max_disp * (gap_size + 1):
            return

        score_candidates = [score for score in [start_row['score'], end_row['score']] if score is not None]
        score = None if not score_candidates else min(score_candidates)
        for offset in range(1, gap_size + 1):
            ratio = offset / (gap_size + 1)
            row = self._pending[start_index + offset]['row']
            row['x'] = round(start_row['x'] + (end_row['x'] - start_row['x']) * ratio, 3)
            row['y'] = round(start_row['y'] + (end_row['y'] - start_row['y']) * ratio, 3)
            row['visible'] = True
            row['score'] = None if score is None else round(score, 6)
            row['_filled'] = True

    def _apply_smoothing(self):
        prev_xy = None if self._last_smoothed_xy is None else self._last_smoothed_xy.copy()
        prev_velocity = None if self._last_velocity_xy is None else self._last_velocity_xy.copy()
        for entry in self._pending:
            row = entry['row']
            if not row['visible']:
                entry['smoothed_xy'] = None
                continue

            xy = np.array([float(row['x']), float(row['y'])], dtype=np.float32)
            effective_score = None if row.get('_filled') else row['score']
            if prev_xy is not None and self._max_jump_disp > 0:
                disp_xy = xy - prev_xy
                jump_disp = np.linalg.norm(disp_xy)
                allowed_jump = self._max_jump_disp
                if prev_velocity is not None:
                    prev_speed = np.linalg.norm(prev_velocity)
                    allowed_jump = max(allowed_jump, prev_speed * self._adaptive_jump_scale + self._adaptive_jump_bias)
                    if prev_speed >= self._prediction_min_speed:
                        predicted_xy = prev_xy + prev_velocity
                        pred_error = np.linalg.norm(xy - predicted_xy)
                        allowed_pred_error = self._prediction_error_base + prev_speed * self._prediction_error_scale
                        if pred_error > allowed_pred_error and (
                            effective_score is None or effective_score < self._motion_score_threshold
                        ):
                            row['visible'] = False
                            row['x'] = None
                            row['y'] = None
                            if row.get('_filled'):
                                row['score'] = None
                            entry['smoothed_xy'] = None
                            continue
                if jump_disp > allowed_jump and (
                    effective_score is None or effective_score < self._max_jump_score_threshold
                ):
                    row['visible'] = False
                    row['x'] = None
                    row['y'] = None
                    if row.get('_filled'):
                        row['score'] = None
                    entry['smoothed_xy'] = None
                    continue
            if prev_xy is None:
                smoothed_xy = xy
            else:
                smoothed_xy = self._smooth_alpha * xy + (1.0 - self._smooth_alpha) * prev_xy
            entry['smoothed_xy'] = smoothed_xy
            prev_velocity = None if prev_xy is None else smoothed_xy - prev_xy
            prev_xy = smoothed_xy

    def _flush(self, final):
        keep = 0 if final else self._interpolate_max_gap + 1
        finalized = []
        while len(self._pending) > keep:
            entry = self._pending.pop(0)
            row = dict(entry['row'])
            row.pop('_filled', None)
            smoothed_xy = entry['smoothed_xy']
            if row['visible'] and smoothed_xy is not None:
                prev_committed_xy = None if self._last_smoothed_xy is None else self._last_smoothed_xy.copy()
                row['x'] = round(float(smoothed_xy[0]), 3)
                row['y'] = round(float(smoothed_xy[1]), 3)
                self._last_smoothed_xy = smoothed_xy
                self._last_velocity_xy = None if prev_committed_xy is None else smoothed_xy - prev_committed_xy
            else:
                row['x'] = None
                row['y'] = None
            finalized.append((row, entry['frame']))
        return finalized


class RealtimeInferenceRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._device = cfg['runner']['device']
        self._frames_in = cfg['model']['frames_in']
        self._out_scales = cfg['model']['out_scales']
        self._input_wh = (cfg['model']['inp_width'], cfg['model']['inp_height'])
        self._output_wh = (cfg['model']['out_width'], cfg['model']['out_height'])
        self._rgb_diff = cfg['model']['rgb_diff']
        self._source = _parse_source(cfg['runner']['source'])
        self._start_frame = int(cfg['runner']['start_frame'])
        self._stride = int(cfg['runner']['stride'])
        self._display = bool(cfg['runner']['display'])
        self._display_wait_ms = int(cfg['runner']['display_wait_ms'])
        self._log_interval = int(cfg['runner']['log_interval'])
        self._max_frames = int(cfg['runner']['max_frames'])
        self._save_video = bool(cfg['runner']['save_video'])
        self._draw_detection = bool(cfg['runner']['draw_detection'])
        self._print_result = bool(cfg['runner']['print_result'])
        self._vis_resize = float(cfg['runner']['vis_resize'])
        self._output_fps = float(cfg['runner']['output_fps'])
        self._trajectory_smoothing_alpha = float(cfg['runner']['trajectory_smoothing_alpha'])
        self._interpolate_max_gap = int(cfg['runner']['interpolate_max_gap'])
        self._interpolate_max_disp = float(cfg['runner']['interpolate_max_disp'])
        self._interpolate_min_score = float(cfg['runner']['interpolate_min_score'])
        self._max_track_jump = float(cfg['runner']['max_track_jump'])
        self._max_jump_score_threshold = float(cfg['runner']['max_jump_score_threshold'])
        self._adaptive_jump_scale = float(cfg['runner']['adaptive_jump_scale'])
        self._adaptive_jump_bias = float(cfg['runner']['adaptive_jump_bias'])
        self._prediction_min_speed = float(cfg['runner']['prediction_min_speed'])
        self._prediction_error_base = float(cfg['runner']['prediction_error_base'])
        self._prediction_error_scale = float(cfg['runner']['prediction_error_scale'])
        self._motion_score_threshold = float(cfg['runner']['motion_score_threshold'])
        self._csv_path = cfg['runner']['output_csv_path']
        self._jsonl_path = cfg['runner']['output_jsonl_path']
        self._video_path = cfg['runner']['output_video_path']

        if self._stride <= 0:
            raise ValueError('runner.stride must be positive')
        if self._start_frame < 0:
            raise ValueError('runner.start_frame must be non-negative')
        if self._vis_resize <= 0:
            raise ValueError('runner.vis_resize must be positive')
        if not (0 < self._trajectory_smoothing_alpha <= 1):
            raise ValueError('runner.trajectory_smoothing_alpha must be in (0, 1]')
        if self._interpolate_max_gap < 0:
            raise ValueError('runner.interpolate_max_gap must be non-negative')
        if self._interpolate_min_score < 0:
            raise ValueError('runner.interpolate_min_score must be non-negative')
        if self._max_track_jump < 0:
            raise ValueError('runner.max_track_jump must be non-negative')
        if self._max_jump_score_threshold < 0:
            raise ValueError('runner.max_jump_score_threshold must be non-negative')
        if self._adaptive_jump_scale < 0:
            raise ValueError('runner.adaptive_jump_scale must be non-negative')
        if self._adaptive_jump_bias < 0:
            raise ValueError('runner.adaptive_jump_bias must be non-negative')
        if self._prediction_min_speed < 0:
            raise ValueError('runner.prediction_min_speed must be non-negative')
        if self._prediction_error_base < 0:
            raise ValueError('runner.prediction_error_base must be non-negative')
        if self._prediction_error_scale < 0:
            raise ValueError('runner.prediction_error_scale must be non-negative')
        if self._motion_score_threshold < 0:
            raise ValueError('runner.motion_score_threshold must be non-negative')

        if self._csv_path is None:
            self._csv_path = f'{self._output_dir}/realtime_coords.csv'
        if self._jsonl_path is None:
            self._jsonl_path = f'{self._output_dir}/realtime_coords.jsonl'
        if self._video_path is None:
            self._video_path = f'{self._output_dir}/realtime_vis.mp4'

        mkdir_if_missing(self._output_dir)
        mkdir_if_missing(osp.dirname(self._csv_path) if osp.dirname(self._csv_path) else self._output_dir)
        mkdir_if_missing(osp.dirname(self._jsonl_path) if osp.dirname(self._jsonl_path) else self._output_dir)
        if self._save_video:
            mkdir_if_missing(osp.dirname(self._video_path) if osp.dirname(self._video_path) else self._output_dir)

        _, self._transform = build_img_transforms(cfg)
        self._detector = build_detector(cfg)
        self._tracker = build_tracker(cfg)

    def _build_affine_mats(self, frame_rgb):
        trans_input = get_transform(frame_rgb, self._input_wh)
        affine_mats = {}
        out_w, out_h = self._output_wh
        for scale in self._out_scales:
            trans_output_inv = get_transform(frame_rgb, (out_w, out_h), inv=1)
            affine_mats[scale] = torch.tensor(
                np.expand_dims(trans_output_inv, axis=0),
                dtype=torch.float32,
            )
            out_w //= 2
            out_h //= 2
        return trans_input, affine_mats

    def _build_input_tensor(self, frames_bgr):
        first_rgb = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB)
        trans_input, affine_mats = self._build_affine_mats(first_rgb)
        imgs_t = []
        for frame_bgr in frames_bgr:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            warped = cv2.warpAffine(frame_rgb, trans_input, self._input_wh, flags=cv2.INTER_LINEAR)
            img = Image.fromarray(warped)
            imgs_t.append(self._transform(img))

        if self._rgb_diff:
            if len(imgs_t) != 2:
                raise ValueError('rgb_diff=True supported only with 2 frames')
            imgs_t[0] = torch.abs(imgs_t[1] - imgs_t[0])

        imgs_t = torch.cat(imgs_t, dim=0).unsqueeze(0)
        return imgs_t, affine_mats

    def _open_writer(self, frame_shape, fps):
        height, width = frame_shape[:2]
        return cv2.VideoWriter(
            self._video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )

    def _render_frame(self, frame, row):
        vis_frame = frame.copy()
        if self._draw_detection:
            x = 0.0 if row['x'] is None else float(row['x'])
            y = 0.0 if row['y'] is None else float(row['y'])
            vis_frame = draw_frame(
                vis_frame,
                center=Center(
                    is_visible=bool(row['visible']),
                    x=x,
                    y=y,
                ),
                color=(0, 255, 255),
                radius=8,
            )
            text = f"frame={row['frame_id']} x={row['x']} y={row['y']} vis={int(row['visible'])}"
            cv2.putText(
                vis_frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if self._vis_resize != 1.0:
            vis_frame = cv2.resize(
                vis_frame,
                None,
                fx=self._vis_resize,
                fy=self._vis_resize,
                interpolation=cv2.INTER_LINEAR,
            )
        return vis_frame

    def _emit_entry(self, row, frame, fps, csv_writer, jsonl_file, video_writer):
        csv_writer.writerow(row)
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        if self._print_result:
            print(json.dumps(row, ensure_ascii=False), flush=True)

        vis_frame = self._render_frame(frame, row)
        if self._save_video:
            if video_writer is None:
                video_writer = self._open_writer(vis_frame.shape, fps)
            video_writer.write(vis_frame)

        should_stop = False
        if self._display:
            cv2.imshow('WASB-SBDT Realtime', vis_frame)
            key = cv2.waitKey(self._display_wait_ms) & 0xFF
            if key in [27, ord('q')]:
                should_stop = True
        return video_writer, should_stop

    @torch.no_grad()
    def run(self):
        capture = cv2.VideoCapture(self._source)
        if not capture.isOpened():
            raise RuntimeError(f'cannot open source: {self._source}')
        if self._start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = self._output_fps

        csv_file = open(self._csv_path, 'w', newline='')
        jsonl_file = open(self._jsonl_path, 'w')
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=['frame_id', 'timestamp_sec', 'x', 'y', 'visible', 'score'],
        )
        csv_writer.writeheader()

        video_writer = None
        recent_frames = deque(maxlen=self._frames_in)
        recent_ids = deque(maxlen=self._frames_in)
        next_window_start = self._start_frame
        last_emitted_frame_id = self._start_frame - 1
        frame_id = self._start_frame
        num_emitted = 0
        trajectory_postprocessor = TrajectoryPostprocessor(
            smooth_alpha=self._trajectory_smoothing_alpha,
            interpolate_max_gap=self._interpolate_max_gap,
            interpolate_max_disp=self._interpolate_max_disp,
            interpolate_min_score=self._interpolate_min_score,
            max_jump_disp=self._max_track_jump,
            max_jump_score_threshold=self._max_jump_score_threshold,
            adaptive_jump_scale=self._adaptive_jump_scale,
            adaptive_jump_bias=self._adaptive_jump_bias,
            prediction_min_speed=self._prediction_min_speed,
            prediction_error_base=self._prediction_error_base,
            prediction_error_scale=self._prediction_error_scale,
            motion_score_threshold=self._motion_score_threshold,
        )

        try:
            self._tracker.refresh()
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                recent_frames.append(frame)
                recent_ids.append(frame_id)

                should_stop = False
                if len(recent_frames) == self._frames_in and recent_ids[0] == next_window_start:
                    frames_window = list(recent_frames)
                    ids_window = list(recent_ids)
                    imgs_t, affine_mats = self._build_input_tensor(frames_window)
                    batch_results, _ = self._detector.run_tensor(imgs_t, affine_mats)
                    frame_results = batch_results.get(0, {})

                    for output_index in sorted(frame_results.keys()):
                        current_frame_id = ids_window[output_index]
                        if current_frame_id <= last_emitted_frame_id:
                            continue

                        track_result = self._tracker.update(frame_results[output_index])
                        timestamp_sec = current_frame_id / fps if fps > 0 else None
                        score = _sanitize_score(track_result['score'])
                        row = {
                            'frame_id': current_frame_id,
                            'timestamp_sec': None if timestamp_sec is None else round(timestamp_sec, 6),
                            'x': None if not track_result['visi'] else round(float(track_result['x']), 3),
                            'y': None if not track_result['visi'] else round(float(track_result['y']), 3),
                            'visible': bool(track_result['visi']),
                            'score': None if score is None else round(score, 6),
                        }
                        finalized_entries = trajectory_postprocessor.push(row, frames_window[output_index])
                        for finalized_row, finalized_frame in finalized_entries:
                            video_writer, stop_requested = self._emit_entry(
                                finalized_row,
                                finalized_frame,
                                fps,
                                csv_writer,
                                jsonl_file,
                                video_writer,
                            )
                            last_emitted_frame_id = finalized_row['frame_id']
                            num_emitted += 1
                            if self._log_interval > 0 and num_emitted % self._log_interval == 0:
                                log.info('processed frames=%d, emitted=%d', frame_id + 1, num_emitted)
                            if stop_requested or (self._max_frames > 0 and num_emitted >= self._max_frames):
                                should_stop = True
                                break
                        if should_stop:
                            break

                    next_window_start += self._stride

                frame_id += 1
                if should_stop:
                    break

            if not (self._max_frames > 0 and num_emitted >= self._max_frames):
                for finalized_row, finalized_frame in trajectory_postprocessor.finish():
                    video_writer, stop_requested = self._emit_entry(
                        finalized_row,
                        finalized_frame,
                        fps,
                        csv_writer,
                        jsonl_file,
                        video_writer,
                    )
                    last_emitted_frame_id = finalized_row['frame_id']
                    num_emitted += 1
                    if self._log_interval > 0 and num_emitted % self._log_interval == 0:
                        log.info('processed frames=%d, emitted=%d', frame_id + 1, num_emitted)
                    if stop_requested or (self._max_frames > 0 and num_emitted >= self._max_frames):
                        break
        finally:
            capture.release()
            csv_file.close()
            jsonl_file.close()
            if video_writer is not None:
                video_writer.release()
            if self._display:
                cv2.destroyAllWindows()

        log.info('realtime inference finished: source=%s emitted=%d csv=%s jsonl=%s video=%s',
                 self._source,
                 num_emitted,
                 self._csv_path,
                 self._jsonl_path,
                 self._video_path if self._save_video else None)
