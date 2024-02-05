import sys
import os
import math
import re
from datetime import datetime
from enum import Enum
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from typing import List, Tuple

import msvcrt

import cv2
import copy
import openpose_modules as op

#from fine_form_net import FineFormNet
#from fine_form_config import FineFormConfig

MSV_KEY_A = b'a'
MSV_KEY_C = b'c'
MSV_KEY_P = b'p'
MSV_KEY_E = b'e'
MSV_KEY_Q = b'q'
MSV_KEY_S = b's'
C = FineFormConfig()


class CaptureStatus(Enum):
    NO_FRAME = -1
    NOT_CAPTURED = 0
    CAPTURED = 1


def _op_empty() -> np.ndarray:
    d = np.zeros([18, 2])
    d[:] = np.nan
    return d


def _captured(stat: CaptureStatus):
    return stat == CaptureStatus.CAPTURED


def keypress_p():
    return msvcrt.getch() != MSV_KEY_P


def keypress_q():
    return msvcrt.getch() != MSV_KEY_Q


def until_quit_key():
    return not cv2.waitKey(1) == ord('q')


def _sys_exit_cause_camera_unused():
    print('カメラまたは動画が見つかりません。または動画エンコーダーが対応していません')
    sys.exit(1)


def _cv2_destroy():
    try:
        cv2.destroyAllWindows()
    except RuntimeError as e:
        print(e)


def _start_video_capturing(path_or_device_id: int or str):
    cap = cv2.VideoCapture(path_or_device_id)
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))

    if not cap.isOpened():
        _sys_exit_cause_camera_unused()

    _, frame = cap.read()
    if frame is None:
        _sys_exit_cause_camera_unused()

    h, w = frame.shape[:2]
    WIDTH = 320  # 解析に十分なサイズがあれば良い

    return cap, int(w * (WIDTH / w)), int(h * (WIDTH / w))


def _norm2d(p1, p2) -> float:
    """座標間の距離：X^2 + Y^2 = Z^2"""
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))


class FormExtractor:
    """
    フォーム抽出器

    動画（ファイルまたはカメラ）からスイングを抽出し、解析用データとして保存する。
    一度読み込んだ動画はquit_capturingで中止するまで処理を継続することができる。

    Parameters:
        video_path: 動画ファイルパス。なければカメラ読み取り。
        save_swing_frames: 抽出したスイングのフレーム画像を保存するのであればTrue。
        skip_frames: スイング開始の検出間隔(フレーム単位)
    """

    def __init__(self, video_path: str = None, skip_frames: int = 8):
        self.__is_camera = video_path is None
        self.__path = video_path
        self.__skip_frames = skip_frames
        cap, w, h = _start_video_capturing(0 if self.__is_camera else self.__path)
        self.__cap = cap
        self.__width, self.__height = w, h
        self.__show_title = video_path
        self.__frame_count = 0
        self.__BASE_COUNT_THRES = 20
        self.__frame_count_thres = self.__BASE_COUNT_THRES

    def __exit__(self):
        self.quit_capturing()

    def quit_capturing(self):
        """
        読込み中のカメラまたは動画を破棄する
        """
        _cv2_destroy()
        self.__cap.release()

    def capture_swing(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        動画のスイングを抽出しデータ化して返却する
        Returns:
            Tuple(s_data, s_frames, vsize)
            s_data: スイングデータ
            s_frames: スイングデータの画像
            vsize: 身体の縦サイズ・ピクセル単位
            success: 取得可否
        """
        def __failure_result():
            return np.array([]), np.array([])

        self.__frame_count_thres += 20
        print('スイング開始を検出します\n全体が映るように構えてください')
        cap_stat, vsize = self.__progress_frame_to_swing_start()

        if _captured(cap_stat):
            print('スイングしてください\n(Q:読込み中断)')
            frames, cap_stat = self.__read_entire_swing(vsize)
            print('frame len', len(frames))
        else:
            print('スイング開始を検出できませんでした')
            frames = np.array([])

        if _captured(cap_stat) and len(frames) > C.input_sequence_len:
            print('抽出を開始します')
            s_data, s_frames, cap_stat = self.__extract_batting_form_data(frames)
        elif _captured(cap_stat) and len(frames) <= C.input_sequence_len:
            print('スイングデータが不足しています')
            cap_stat = CaptureStatus.NOT_CAPTURED
        else:
            print('スイング終了を検出できませんでした')

        if _captured(cap_stat):
            # 読み取りが完了するまでは次第に長く読む
            # 完了したらリセット
            self.__frame_count_thres = self.__BASE_COUNT_THRES
        else:
            print('抽出に失敗しました')
            s_data, s_frames = __failure_result()

        return s_data, s_frames, vsize, cap_stat

    def __cv2_imshow(self, img: np.ndarray, t=1):
        cv2.imshow(self.__show_title, img)
        cv2.waitKey(t)

    def __cap_back_frames(self, back: int = 20):
        self.__frame_count -= back
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, self.__frame_count)

    def __cap_read(self):
        ret, frame = self.__cap.read()
        self.__frame_count += 1
        if frame is not None:
            frame = cv2.resize(frame, dsize=(self.__width, self.__height))
        return ret, frame

    def __landmarks_is_swing_start(self, lm, lift_bat_thres=0.14):
        """各部位からスイング開始フォームの判定を行う"""
        head = lm[C.lg_head][0]
        legs = lm[C.lg_r_ankle][0]
        r_wrist, l_wrist = lm[C.lg_wrist]
        r_midhip, l_midhip = lm[C.lg_midhip]
        # 部位のピクセル位置を身体サイズに対する比率としてデータ化する
        # 基準は頭部と脚部のピクセル数とする
        vsize = float(abs(head[1] - legs[1]))
        _dist_wrists = _norm2d(r_wrist, l_wrist) / vsize
        _seal_other_hand = np.any(np.isnan([r_wrist, l_wrist])) and not np.all(np.isnan([r_wrist, l_wrist]))
        near_wrists = _dist_wrists < 0.4 or _seal_other_hand
        _finite_wrist = r_wrist if np.isfinite(r_wrist[1]) else l_wrist
        _dist_mh_wr = (r_midhip[1] - _finite_wrist[1]) / vsize
        rside_wr = head[0] > _finite_wrist[0]
        _dist_hd_wr = (_finite_wrist[1] - head[1]) / vsize
        vnear_head_wr = _dist_hd_wr < 0.24
        lift_bat = _dist_mh_wr > lift_bat_thres
        _dist_lr_mhi = (l_midhip[0] - r_midhip[0]) / vsize
        flat_mhi = _dist_lr_mhi > 0.12
        whole_body = np.all(np.isfinite(lm[C.lg_all_ff]))
        swing_start = near_wrists and rside_wr and vnear_head_wr and lift_bat and flat_mhi and whole_body
        print('S', swing_start, 'near-wr', f'{_dist_wrists:.3f}', near_wrists,
              'rside-wr', rside_wr,
              'vhead-wr', f'{_dist_hd_wr:.3f}', vnear_head_wr,
              'lift', f'{_dist_mh_wr:.3f}',
              'flat', f'{_dist_lr_mhi:.3f}',
              'in', whole_body)
        return swing_start, vsize

    def __landmarks_is_swing_end(self, lm, vsize):
        # 各部位からスイング終了フォームの判定
        _, l_wrist = lm[C.lg_wrist]
        body, _, l_sholder = lm[C.lg_upper]
        _dist_bd_wr = _norm2d(body, l_wrist) / vsize
        far_body_wrist = _dist_bd_wr > 0.1
        _lside_wr = (l_wrist[0] - l_sholder[0]) / vsize
        leftside_wrist = _lside_wr > 0.1
        whole_body = np.all(np.isfinite(C.lg_all_ff))
        swing_end = far_body_wrist and leftside_wrist and whole_body
        print('E', swing_end, 'body-wrist', f'{_dist_bd_wr:.3f}',
              'lside-wrist(x)', f'{_lside_wr:.3f}',
              'in', whole_body)
        return swing_end

    def __progress_frame_to_swing_start(self) -> Tuple[bool, float]:
        """スイング開始までフレームを進める"""
        frame_count = 0
        skip = self.__skip_frames
        lift_bat_thres = 0.1

        def _adjust_lift_thresholds():
            nonlocal lift_bat_thres
            if lift_bat_thres > 0.1:
                lift_bat_thres -= 0.01

        def _until_limit_camera_capturing():
            return not self.__is_camera or (self.__is_camera and frame_count < 500)

        vsize = -1
        cap_stat = CaptureStatus.NOT_CAPTURED
        while _until_limit_camera_capturing() and until_quit_key():
            _, frame = self.__cap_read()
            if frame is None:
                cap_stat = CaptureStatus.NO_FRAME
                break
            frame_count += 1
            if frame_count % skip != 0:
                continue
            candidate, subset = op.body_estimation(frame)
            # print('cnad', candidate)
            # print('subs', subset)
            if subset is None or len(subset) == 0:
                self.__cv2_imshow(frame)
                continue
            lm = extract_landmarks_or_nan(candidate, subset)
            # print('lmarks', lm)
            canvas = copy.deepcopy(frame)
            canvas = op.op_util.draw_bodypose(canvas, candidate, subset)
            self.__cv2_imshow(canvas)
            # cv2.imshow('Form Input', frame)
            is_start, vsize = self.__landmarks_is_swing_start(lm, lift_bat_thres)
            if is_start:
                print('スイング開始を検知')
                cap_stat = CaptureStatus.CAPTURED
                break
            elif frame_count > 100:
                # 検知が遅ければ閾値を下げる
                _adjust_lift_thresholds()

        return cap_stat, vsize

    def __read_entire_swing(self, vsize: float):
        """スイング全体を読みこむ・終端位置で停止・一定フレーム経過で取り直し"""
        frame_count = 0
        steps = C.step_swing_capturing

        if self.__is_camera:
            def _until_limit_camera_capturing():
                return frame_count < 500
        else:
            def _until_limit_camera_capturing():
                return True

        # XXX: キャプチャしながら姿勢推定は処理が重い。最初は固定時間で取得して後から終了判定が良いかも
        frames = []
        cap_stat = CaptureStatus.NOT_CAPTURED
        while _until_limit_camera_capturing() and until_quit_key():
            _, frame = self.__cap_read()
            if frame is None:
                cap_stat = CaptureStatus.NO_FRAME
                break
            frame_count += 1
            frames.append(frame)
            self.__cv2_imshow(frame)
            if frame_count % steps != 0:
                continue

            candidate, subset = op.body_estimation(frame)
            if len(subset) == 0:
                continue

            lm = extract_landmarks_or_nan(candidate, subset)
            is_end = self.__landmarks_is_swing_end(lm, vsize)
            if is_end:
                print('スイング終了')
                cap_stat = CaptureStatus.CAPTURED
                break
            elif not self.__is_camera and frame_count > self.__frame_count_thres:
                # 動画では開始が遅れる場合が多いので
                # 遅れた場合は少し戻して多めにフレームを取り直す
                is_start, _ = self.__landmarks_is_swing_start(lm)
                if is_start:
                    print('開始を再検出・フレームを取り直します')
                    self.__cap_back_frames(self.__frame_count_thres / 2 + 8)  # 8=step
                    return np.array([]), CaptureStatus.NOT_CAPTURED

        if _captured(cap_stat):
            # 振り終わりのフォロースルーを得るため
            # 終了検知までにかかったフレーム数に応じて余分にフレームを読む
            for _ in range(math.ceil(len(frames) / 20) + 2):
                _, frame = self.__cap_read()
                if frame is None:
                    break
                frames.append(frame)

        return np.array(frames), cap_stat

    def __extract_batting_form_data(self, frames: np.ndarray) -> np.ndarray:
        """
        フレームからバッティングフォームを抽出する。

        Args:
            frames: スイングデータ列 shape[L, ...Image_Shape] (L:取得フレーム数)
        Returns:
            以下のTuple。(N:抽出スイング数)
            * 抽出後のデータ shape[N, 18, 2]
            * 抽出後のフレーム shape[N, ...Image_Shape]
        """
        def __gain_body_data(s_idx: int):
            frame = frames[s_idx]
            candidate, subset = op.body_estimation(frame)
            return extract_landmarks_or_nan(candidate, subset), frame

        def __valid_ff_data(data: np.ndarray):
            return np.all(np.isfinite(data[C.lg_all_ff]))

        def __find_finite(s: int, e: int) -> Tuple[np.ndarray, np.ndarray]:
            for j in range(s, e):
                d, f = __gain_body_data(j)
                if d is not None and __valid_ff_data(d):
                    print('found')
                    return d, f
            print('not found')
            return None, None

        def __find_prev_finite(i: int) -> Tuple[np.ndarray, np.ndarray]:
            s = i - 3 if i - 3 >= 0 else 0
            print(f'find prev {s} to {i}')
            return __find_finite(s, i)

        def __find_next_finite(i: int) -> Tuple[np.ndarray, np.ndarray]:
            e = i + 3 if i + 3 < len(frames) else len(frames) - 1
            print(f'find next {e} to {i}')
            return __find_finite(i, e)

        s_data = []
        s_frames = []
        s_indices = slice_for_ff_input(frames)

        def __failure_result():
            return s_data, s_indices, CaptureStatus.NOT_CAPTURED

        print('sliced', s_indices.shape)
        if len(s_indices) != C.input_sequence_len + 1:
            print(f'warning: sliced data length is {len(s_indices)}, not {C.input_sequence_len + 1}')
            return __failure_result()

        # 端データにNaNがあると正常なデータといえないため、周辺で取り直す
        s_idx_s, s_idx_e = s_indices[0], s_indices[-1]
        d0, f0 = __gain_body_data(s_idx_s)
        de, fe = __gain_body_data(s_idx_e)
        if d0 is None or not __valid_ff_data(d0):
            d, f = __find_prev_finite(s_idx_s)
            d0 = d0 if d is None else d
            f0 = f0 if f is None else f
        if de is None or not __valid_ff_data(de):
            d, f = __find_next_finite(s_idx_e)
            de = de if d is None else d
            fe = fe if f is None else f
        if d0 is None or de is None:
            return __failure_result()

        s_data.append(d0)
        s_frames.append(f0)
        for i in range(1, C.input_sequence_len):
            s_idx = s_indices[i]
            d, f = __gain_body_data(s_idx)
            if d is None:
                return __failure_result()
            s_data.append(d)
            s_frames.append(f)
        s_data.append(de)
        s_frames.append(fe)

        for f in s_frames:
            self.__cv2_imshow(f)

        return np.array(s_data), np.array(s_frames), CaptureStatus.CAPTURED


def slice_for_ff_input(frames: np.ndarray) -> List[int]:
    """
    スイングフレームのスライシング
    判定モジュール入力用に、幾つかのフレームを選別して取得する。
    スイング後半ほど速いので取得間隔は漸次的に短くする。

    Returns:
        スイングデータのインデックスリスト Shape[N + 1]
    """
    N = C.input_sequence_len
    base_skip = math.ceil(len(frames) / (N + 1))
    e_skip = math.ceil(base_skip / 8)  # 最小1は確保

    skip = e_skip
    skip_inc = (base_skip - e_skip) / (N + 1)
    inc_coef = 1 / (N + 1)
    print(skip, skip_inc)

    # 計算のしやすさから末尾から辿る
    # もともとスイング終了後のフレームも余分に渡してある想定なので少し余して取る(調整値は経験則)
    index = math.floor(-0.1 * len(frames))
    s_frames_idx = []
    for i in range(N + 1):
        s_frames_idx.append(index)
        print(i, index, skip)
        skip += skip_inc * inc_coef * i
        index = index - math.floor(skip)

    # 外部での利用では正数インデックスのほうが扱いやすいため直す
    return np.array([len(frames) + i for i in reversed(s_frames_idx)])


def nan_xy(): return np.array([np.nan, np.nan])


def extract_landmarks_or_nan(cand, subs) -> np.ndarray:
    """
    18個の部位について、あればその座標、なければNaNで返却。
    複数の検知があった場合、最大サイズの検出を採用する。

    Parameters:
        cand: OpenPoseの返却値・取得できた部位の座標及びスコア。
            Shape[N, 4] (N=検出された部位数)
        subs: OpenPoseの返却値・部位の検出有無。検出された個体の数が長さとなる。
            Shape[D, 20] (D=検出された個体数, D<N, 部位インデックス*18 + 全体スコア*1 + 個体検出位置インデックス)
    Returns:
        最大サイズの部位データ(1:頭,2:首,...)
        最大サイズは鼻(0)と右足首(10)の距離とする。
        ndarray shape[18, 2]
        [
            [x1, y1],
            [x2, y2],
            ...
            [x18, y18],
        ]
    """
    if cand is None or subs is None or len(subs) == 0:
        return None
    h, l = C.lg_head[0], C.lg_leg[1]
    # 最大サイズのsubsetを見つける
    max_i = np.argmax([abs(cand[int(s[h]), 1] - cand[int(s[l]), 1]) for s in subs])
    # subsetの指定するcandidateの(x,y)を取得
    return np.array([cand[int(x), :2] if x != -1 else nan_xy() for x in subs[max_i, :18]])


def convert_landmarks_to_input_data(data: np.ndarray, vsize: float = None) -> np.ndarray:
    """
    身体データをフォーム判定入力に変換
    身体データからフォーム判定入力で使用する部位のみ取得し、移動距離を移動比率に変換する

    Parameters:
        data: Shape[N+1, 18, 2] スイングデータ・時系列に並んだN+1個のOpenPoseの身体部位座標
            Nはフォーム判定入力の時系列サイズ。
            先頭データは基準値として用いるため、変換後の値からは除外される。
        vsize: (Optional)頭部(鼻)と足首の間のy軸方向ピクセル数。なければ0番目のデータから読み取る。
    Returns:
        変換後のデータ: Shape[N*P*2] 判定で用いる部位の移動比率(P:判定部位の個数)
            但しデータの有効性が必要なため、先頭データに鼻と足が含まれていなければNoneを返す。
    """
    # assert np.all(np.isfinite(data[0, C.lg_vbody])), f'先頭データの鼻と右足首が未検出です, data={data}'
    if not np.all(np.isfinite(data[0, C.lg_vbody])):
        return None

    if vsize is None:
        nose, rakl = data[0, C.lg_vbody]
        vsize = abs(nose[1] - rakl[1])

    # フォーム判定で使用する箇所を抽出
    s_data = copy.deepcopy(data)[:, C.lg_all_ff]
    # 基準位置を0として基準位置からの移動距離に直す
    # 先頭データは基準位置(すべて0)なので除外
    s_data = s_data[1:] - s_data[0, :]
    # 移動距離を、身体の縦サイズとの移動比率に変換
    s_data = s_data / vsize
    # 検出されなかった部位の値NaNを補間する
    # 間のNaNは中間値で、端のNaNは隣接値で埋める
    s_data = s_data.reshape([s_data.shape[0], math.prod(s_data.shape[1:])])
    s_data = interpolate_points(s_data)
    # 入力サイズ = 時系列長 * 使用部位 * 2
    return s_data.reshape(-1)


def interpolate_points(points: np.ndarray) -> np.ndarray:
    """座標データを補間・NaNの代替値を埋める"""
    return pd.DataFrame(points).interpolate(limit_direction='forward').values


def frames_dir_to_input(file_path: os.PathLike):
    """
    画像ディレクトリから順次画像で姿勢推定し、フォーム判定モジュールの入力データを作成する。

    Parameters:
        file_path: パス(ファイルまたはディレクトリ)
    """
    N = C.input_sequence_len

    # ファイルは辞書順に時系列で並んでいるものとする
    files = sorted(os.listdir(file_path))
    assert len(files) >= N + 1, f'ファイル数が不足しています: {len(files)} < {N} + 1'

    # seq_len + 1 だけファイルを読み込んで時系列データにする
    frames = [cv2.imread(os.path.join(file_path, f)) for f in files[:N+1]]
    data = collect_ff_points(frames, quiet=True)
    return convert_landmarks_to_input_data(data)


def collect_ff_points(frames: np.ndarray, quiet: bool = False) -> np.ndarray:
    """画像データを、フォーム判定で用いる部位の、時系列座標データに変換"""
    N = C.input_sequence_len
    # 部位座標の格納先を作っておく。欠損値を補間するため
    # 基準データ[0]と時系列データ[1:seq_len+1]があるので、時系列サイズ＋１
    data = np.empty([N + 1, len(C.lg_all_ff), 2])

    def __print_indetected(points):
        nans = np.argwhere(np.isnan(points))
        nans = np.unique(nans[:, 0]) if len(nans) > 0 else []
        if not quiet:
            print('no detected parts', C.get_labels(nans))

    for i, img in enumerate(frames):
        candidates, subset = op.body_estimation(img)
        if not quiet:
            print(f'body --- {i:02d} ---')
            # print('candidate', candidates)
            print('subset', subset)
        points = extract_landmarks_or_nan(candidates, subset)
        __print_indetected(points)
        data[i, :] = points[C.lg_all_ff]

    return data


def save_form_data(data_name: str, data: np.ndarray, frames: np.ndarray):
    save_images(f'../input_images/{data_name}', frames)
    save_swing_data(f'../data/input/{data_name}.csv', data)


def save_images(img_dir: str, images: np.ndarray):
    """スイング画像を保存"""
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(img_dir, f'{i:02d}.png'), img)
    print(f'フォーム画像を保存しました: {img_dir}')


def save_swing_data(data_file: str, data: np.ndarray):
    """スイングデータ(OpenPoseの18部位x,y)を保存"""
    if len(data.shape) > 2:
        # ３次以上のデータになっている場合行列に直す
        data = data.reshape([data.shape[0], math.prod(data.shape[1:])])
    pd.DataFrame(data, columns=C.ALL_COLUMNS).to_csv(data_file, index=False)
    print(f'フォームデータを保存しました: {data_file}')


def append_ff_input(ff_input: np.ndarray, target_score: float):
    """判定入力データを訓練CSVに追加"""
    with open(f'../data/train.csv', mode='a') as w:
        if ff_input is not None:
            w.write(','.join(np.char.mod('%.6f', ff_input)))
            w.write(f',{target_score}\n')


def run_input_module_main(video_path: str = None, target_score: int = 0, start: int = 0, auto: bool = False,
                          save_images: bool = False, skip_frames: int = 8,
                          analysis_proc = None) -> np.ndarray:
    """
    入力モジュールメイン処理
    * スイング読み取り
    * スイングデータ保存

    Parameters:
        video_path: 動画パス。なければカメラ読み取り
        target_score: 読み取ったスイングデータに付けるスコア
        analysis_proc: 読み取りデータの解析プロセス。なければデータ読込みのみ実行される
    Returns:
        判定モジュール入力 Shape[280] 時系列毎の部位の変化率(10*14*2)
    """
    print(f'解析モード: {"ON" if analysis_proc is not None else "OFF"}')
    data_name = ''  # データ保存名
    img_id = start - 1

    # フォーム抽出器はカメラまたは与えたファイルから連続的にフォームを読み取る
    ex = FormExtractor(video_path, skip_frames=skip_frames)

    if video_path is None:
        data_name = f'form_{datetime.now().strftime("%Y%m%d%H%M%S")}_{target_score}'

        def __run_capturing():
            print('カメラからスイングを読み込みます🈩')
            print('\nカメラは水平にして全身が映るようにしてください')
            return __cap_form()
    else:
        data_name = re.split(r'[/\\]', video_path)[-1].split('.')[0]

        def __run_capturing():
            print('動画からスイングを読み込みます🈩')
            return __cap_form()

    # １回分のフォーム・スイング読み取り
    def __cap_form() -> np.ndarray:
        data, frames, vsize, cap_stat = ex.capture_swing()
        if _captured(cap_stat):
            return convert_landmarks_to_input_data(data, vsize), data, frames, cap_stat
        else:
            return None, None, None, cap_stat

    def __save_data(ff_input, s_data, s_frames, score=None):
        if ff_input is None or s_data is None:
            print('スイングデータがないため保存をキャンセルします')
            return
        nonlocal img_id
        img_id += 1
        scr = '' if score is None or np.isnan(score) else f'_{score:.2f}'
        save_form_data(f'{data_name}{scr}_{img_id:04d}', s_data, s_frames)
        append_ff_input(ff_input, target_score)

    def __wait_action_and_continue(ff_input, s_data, s_frames):
        k = msvcrt.getch()
        score = None
        if k == MSV_KEY_A and analysis_proc is not None:
            score = analysis_proc(ff_input)
        if (k == MSV_KEY_E or k == MSV_KEY_S or k == MSV_KEY_A):
            __save_data(ff_input, s_data, s_frames, score=score)
        return not (k == MSV_KEY_Q or k == MSV_KEY_E)

    ff_input: np.ndarray = None

    while True:
        ff_input, s_data, s_frames, cap_stat = __run_capturing()
        if cap_stat == CaptureStatus.NO_FRAME:
            print('フレーム終端です')
            break

        if auto:
            if save_images:
                __save_data(ff_input, s_data, s_frames)
            continue

        print('次の操作をしてください\n (Q:保存せず終了, E:保存して終了, C:保存せず続行,\n  S:保存して続行, A:判定して続行)')
        if not __wait_action_and_continue(ff_input, s_data, s_frames):
            break

    if ff_input is not None:
        ff_input.put(-1, target_score)

    return ff_input


class InputModuleArgument:
    def __init__(self, parser: ArgumentParser):
        p = parser.parse_args()
        self.target_score = p.eval_score
        self.video_path = p.data
        self.skip = p.skip_frames
        self.start_num = p.start_num
        self.auto = p.auto
        self.save_images = p.save_images


def create_arguments() -> InputModuleArgument:
    """メイン処理のコマンド引数を定義"""
    # py xxx.py -d aaa.mp4 -v 1 -s -a
    p = ArgumentParser()
    p.add_argument('--eval-score', '-v', required=True, type=int, default=0, help='この動画に付けるスコア(0 | 1)')
    p.add_argument('--data', '-d', help='入力動画パス(デフォルト；カメラ入力)')
    p.add_argument('--skip-frames', '-f', type=int, default=15, help='開始判定フレームの取得間隔(デフォルト:15)')
    p.add_argument('--start-num', '-n', type=int, default=0, help='保存ファイル名連番の開始番号(デフォルト:0開始)')
    p.add_argument('--auto', '-a', action='store_true', help='自動実行(デフォルト:False)')
    p.add_argument('--save-images', '-s', action='store_true', help='自動実行時、スイング画像を保存(デフォルト:False)')
    return InputModuleArgument(p)


if __name__ == '__main__':
    # os.makedirs(f'../data/input/', exist_ok=True)
    args = create_arguments()

    # 引数があればそのディレクトリのファイルをデータに変換
    if args.video_path is not None:
        path = args.video_path
        print(f'{path}の画像からフォームデータを生成します')
    else:
        path = None

    run_input_module_main(path, args.target_score, start=args.start_num, auto=args.auto,
                          save_images=args.save_images, skip_frames=args.skip)
    print('\n終了します')
