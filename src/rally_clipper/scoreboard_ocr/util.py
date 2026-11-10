"""Scoreboard OCR over a match video, and the score-sequence and ball-track interval logic the clippers use."""

import os
import pickle

import numpy as np

from rally_clipper.scoreboard_ocr.scoreboard_identifier import TextIdentifier
from rally_clipper.scoreboard_ocr.video_processor import VideoFrameIterator


def process_one(gameplay_dir):
    """OCR the scoreboard every 5th frame (every frame near a change) and append (ok, frame, text) records to sb/sb.pkl."""
    id = os.path.basename(gameplay_dir)  ##*#
    video_file = os.path.join(gameplay_dir, "video", f"{id}.mp4")
    sb_save_dir = os.path.join(gameplay_dir, "sb")
    sb_save_file = os.path.join(sb_save_dir, "sb.pkl")

    print(f"*******************{id}*******************")

    if not os.path.isdir(sb_save_dir):
        os.mkdir(sb_save_dir)

    video_processing = VideoFrameIterator(video_file)
    score_identifier = TextIdentifier()

    buffer_size = 20000
    results_buffer = []
    frame_count = 0
    with open(sb_save_file, 'ab') as file:
        sb_not_found_count = 0

        for frame in video_processing:
            ok, result = score_identifier.call_ocr(frame)

            if not ok and result == "Board not found, skip frame":
                sb_not_found_count += 1
            else:
                # restart the counter
                sb_not_found_count = 0

            results_buffer.append((ok, frame_count, result))
            frame_count += 1

            if len(results_buffer) >= buffer_size:
                pickle.dump(results_buffer, file)
                file.flush()
                results_buffer.clear()

            if sb_not_found_count == 0:
                # we don't need to sample every frame for the purpose of knowing when the scoreboard changes
                for _ in range(4):
                    next(video_processing, None)
                    results_buffer.append((False, frame_count, "OPTIM (intermediate frame), skip frame"))
                    frame_count += 1

            if sb_not_found_count == 100:
                sb_not_found_count = 0
                for _ in range(5 * 30):
                    next(video_processing, None)
                    results_buffer.append((False, frame_count, "OPTIM (sb not found), skip frame"))
                    frame_count += 1

            if frame_count % 10000 == 0:
                print(frame_count)

        if results_buffer:
            pickle.dump(results_buffer, file)
            file.flush()
            results_buffer.clear()

    # signal that the tracking is complete
    with open(f"{sb_save_dir}/done.txt", "w") as _:
        pass


def exit_procedure(results_buffer, file, sb_save_dir):
    """Flush the remaining records and write sb/done.txt."""
    if results_buffer:
        pickle.dump(results_buffer, file)
        file.flush()
        results_buffer.clear()

    # signal that the tracking is complete
    with open(f"{sb_save_dir}/done_second.txt", "w") as _:
        pass


def compute_const_intervals(signal, strength_threshold, time_threshold):
    """
    Identify contiguous intervals in a 1D signal where the magnitude remains
    below a specified strength threshold for longer than a minimum duration.

    This function treats segments where `|signal[t]| <= strength_threshold`
    as "near-zero" regions. It then detects the start and end indices of all
    such regions and returns only those whose length exceeds `time_threshold`
    frames.

    Parameters
    ----------
    signal : array_like
        1D array representing the signal over time.
    strength_threshold : float
        Magnitude threshold used to determine whether the signal is considered
        near zero (i.e., `abs(signal) <= strength_threshold`).
    time_threshold : int
        Minimum interval length (in frames) required for a detected near-zero
        region to be included in the output.

    Returns
    -------
    zero_intervals : list of tuple(int, int)
        A list of `(start, end)` index pairs, where each pair denotes a
        contiguous interval in which the signal stays below `strength_threshold`
        for more than `time_threshold` frames. The intervals are returned with
        `start` inclusive and `end` exclusive, consistent with NumPy indexing.

    Notes
    -----
    - Interval boundaries are detected using edges in a binary mask of
      near-zero frames.
    - `time_threshold` is interpreted in units of frames; convert externally if
      working in seconds.
    """
    near_zero = np.abs(signal) <= strength_threshold
    diff = np.diff(np.concatenate(([0], near_zero.astype(int), [0])))  # Detect edges

    zero_intervals = []
    for start, end in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
        frame_difference = end - start
        # print('frame_difference', frame_difference, time_threshold)
        if frame_difference > time_threshold:
            zero_intervals.append((start, end))

    return zero_intervals


def find_overlapping_intervals(intervals1, intervals2):
    """
    Compute all pairwise overlaps between two sorted lists of 1D intervals.

    Given two lists of non-overlapping, sorted intervals `intervals1` and
    `intervals2`, this function scans both lists in linear time and returns the
    intervals where they intersect. Each interval is of the form `(start, end)`
    with `start < end`.

    Parameters
    ----------
    intervals1 : array_like or list of tuple(int, int)
        First list of intervals, sorted by start time and non-overlapping
        within the list.
    intervals2 : array_like or list of tuple(int, int)
        Second list of intervals, also sorted and non-overlapping within the list.

    Returns
    -------
    overlapping_intervals : np.ndarray of shape (K, 2)
        Array of `(start, end)` pairs representing all intervals where
        `intervals1` and `intervals2` overlap. For each overlapping pair,
        the returned interval is:
            (max(start1, start2), min(end1, end2))

    Notes
    -----
    - The function runs in O(len(intervals1) + len(intervals2)) time by
      advancing the pointer of the interval that ends first.
    - Intervals are assumed to use the same coordinate system and to be
      half-open or fully closed consistently; overlap is checked via:
          end1 > start2 and end2 > start1
    """

    overlapping_intervals = []

    i, j = 0, 0
    while i < len(intervals1) and j < len(intervals2):
        start1, end1 = intervals1[i]
        start2, end2 = intervals2[j]

        # Check for overlap
        if end1 > start2 and end2 > start1:
            overlapping_intervals.append((max(start1, start2), min(end1, end2)))

        # Move to the next interval in the list that ends earlier
        if end1 < end2:
            i += 1
        else:
            j += 1

    return np.array(overlapping_intervals)


def zero_out_intervals(signal, intervals):
    """Zero out values in the given signal within the specified intervals."""
    for start, end in intervals:
        signal[start:end] = 0
    return signal


def find_complementary_intervals(signal, intervals, min_length=0):
    """
    Compute the complementary (non-overlapping) intervals of a 1D signal given
    a set of existing intervals.

    Given a list of intervals inside a signal timeline, this function returns
    all gaps—i.e., regions not covered by any interval—optionally filtering out
    complementary intervals whose length is shorter than `min_length`.

    The function assumes:
    • `intervals` are sorted by their start index,
    • intervals may be adjacent or overlapping,
    • interval boundaries follow a half-open convention `(start, end)`.

    Parameters
    ----------
    signal : array_like
        The underlying 1D signal; only its length is used.
    intervals : list of tuple(int, int)
        List of `(start, end)` intervals to exclude. These intervals should be
        sorted and non-decreasing in their start indices.
    min_length : int, optional (default = 0)
        Minimum allowable length (in frames) for a complementary interval to
        be retained.

    Returns
    -------
    complementary_intervals : list of tuple(int, int)
        List of `(start, end)` intervals representing all regions of the signal
        not covered by any interval in `intervals`. Each interval is half-open
        (`start` inclusive, `end` exclusive).

    Notes
    -----
    - Adjacent or overlapping input intervals are treated as a single covered
      region for the purpose of finding gaps.
    - The function scans linearly through the intervals, tracking uncovered
      segments between `prev_end` and the next `start`.
    - A final complementary interval is added at the end of the signal if
      the last covered interval does not reach the end.
    """
    complementary_intervals = []
    prev_end = 0
    for start, end in intervals:
        if start - 1 <= prev_end:
            prev_end = end + 1
            continue
        length = start - prev_end
        if length < min_length:
            prev_end = end + 1
            continue
        complementary_intervals.append(
            (prev_end, start)  # (inclusive, exclusive)
        )
        prev_end = end + 1

    # OPTIM: HMMM i don't want points where the clip is cut off prematurely
    # no this is too aggressive
    if not len(signal) - 1 <= prev_end:
        complementary_intervals.append((prev_end, len(signal)))

    return complementary_intervals


def best_point_estimate(correlation_x, signal_intervals, minimum_point_length_threshold, corr_x_peak):
    """
    Select the most likely gameplay interval based on peak activity in a
    correlation signal.

    This function evaluates a set of candidate intervals and returns the index
    of the interval that contains the largest number of above-threshold peaks in
    `correlation_x`. Only intervals whose temporal length exceeds
    `minimum_point_length_threshold` are considered.

    Parameters
    ----------
    correlation_x : array_like
        1D correlation signal (e.g., cross-correlation between score digits or
        motion cues) from which peak activity is measured.
    signal_intervals : list of tuple(int, int)
        Candidate `(start, end)` intervals (half-open). These typically represent
        contiguous regions of interest extracted earlier in processing.
    minimum_point_length_threshold : int
        Minimum required interval length (in frames) for an interval to be
        considered a valid candidate.
    corr_x_peak : float
        Absolute-value threshold on the correlation signal. Peaks are counted
        as frames where `abs(correlation_x[t]) > corr_x_peak`.

    Returns
    -------
    best_guess : int
        The index (into `signal_intervals`) of the interval containing the
        highest number of above-threshold peaks. Returns `-1` if no interval
        satisfies the length requirement or no peaks exceed the threshold.

    Notes
    -----
    - Peak count serves as a heuristic for selecting the interval most likely
      corresponding to meaningful gameplay activity.
    - If multiple intervals yield the same number of peaks, the earliest one
      encountered in `signal_intervals` is chosen.
    """
    best_guess = -1
    max_num_corr_x_peaks = 0
    for i, possible_gameplay_interval in enumerate(signal_intervals):
        start, end = possible_gameplay_interval
        frame_difference = end - start
        if frame_difference <= minimum_point_length_threshold:
            # not long enough
            continue

        signal = correlation_x[start:end]
        num_corr_x_peaks = np.count_nonzero(np.abs(signal) > corr_x_peak)

        if num_corr_x_peaks > max_num_corr_x_peaks:
            best_guess = i
            max_num_corr_x_peaks = num_corr_x_peaks

    return best_guess


def is_number(s):
    """True for ints, floats and strings that read as a number."""
    return isinstance(s, (int, float)) or (isinstance(s, str) and s.replace('.', '', 1).isdigit())


def check_format_1(score_tokens):
    """Checks if the tokens are in the format <name> <score1> <score2> <name> <score1> <score2>"""

    if len(score_tokens) != 6:
        return False

    return (
        not is_number(score_tokens[0])
        and is_number(score_tokens[1])
        and is_number(score_tokens[2])
        and not is_number(score_tokens[3])
        and is_number(score_tokens[4])
        and is_number(score_tokens[5])
    )


def check_format_2(score_tokens):
    """Tokens read as <name> <series> <round> <series> <round> <name>."""
    return len(score_tokens) == 6 and (
        not is_number(score_tokens[0])
        and is_number(score_tokens[1])
        and is_number(score_tokens[2])
        and is_number(score_tokens[3])
        and is_number(score_tokens[4])
        and not is_number(score_tokens[5])
    )


def check_format_3(score_tokens):
    """Tokens read as <series> <round> <name> <name> <series> <round>."""
    return len(score_tokens) == 6 and (
        is_number(score_tokens[0])
        and is_number(score_tokens[1])
        and not is_number(score_tokens[2])
        and not is_number(score_tokens[3])
        and is_number(score_tokens[4])
        and is_number(score_tokens[5])
    )


def check_format_4(score_tokens):
    """Tokens read as <name> <name> <series> <round> <name> <series> <round>."""
    return len(score_tokens) == 7 and (
        not is_number(score_tokens[0])
        and not is_number(score_tokens[1])
        and is_number(score_tokens[2])
        and is_number(score_tokens[3])
        and not is_number(score_tokens[4])
        and is_number(score_tokens[5])
        and is_number(score_tokens[6])
    )


def check_format_5(score_tokens):
    """Tokens read as <name> <series> <round> <name> <name> <series> <round>."""
    return len(score_tokens) == 7 and (
        not is_number(score_tokens[0])
        and is_number(score_tokens[1])
        and is_number(score_tokens[2])
        and not is_number(score_tokens[3])
        and not is_number(score_tokens[4])
        and is_number(score_tokens[5])
        and is_number(score_tokens[6])
    )


def check_format_6(score_tokens):
    """Tokens read as <name> <series> <round> <name> <series> <round> <name>."""
    return len(score_tokens) == 7 and (
        not is_number(score_tokens[0])
        and is_number(score_tokens[1])
        and is_number(score_tokens[2])
        and not is_number(score_tokens[3])
        and is_number(score_tokens[4])
        and is_number(score_tokens[5])
        and not is_number(score_tokens[6])
    )


def check_format_7(score_tokens):
    """Tokens read as <name> <name> <name> <series> <round> <name> <name> <series> <round>."""
    return len(score_tokens) == 9 and (
        not is_number(score_tokens[0])
        and not is_number(score_tokens[1])
        and not is_number(score_tokens[2])
        and is_number(score_tokens[3])
        and is_number(score_tokens[4])
        and not is_number(score_tokens[5])
        and not is_number(score_tokens[6])
        and is_number(score_tokens[7])
        and is_number(score_tokens[8])
    )


def check_format_8(score_tokens):
    """Tokens read as <name> <name> <series> <round> <series> <round> <name>."""
    return len(score_tokens) == 7 and (
        not is_number(score_tokens[0])
        and not is_number(score_tokens[1])
        and is_number(score_tokens[2])
        and is_number(score_tokens[3])
        and is_number(score_tokens[4])
        and is_number(score_tokens[5])
        and not is_number(score_tokens[6])
    )


def check_format_9(score_tokens):
    """Tokens read as <series> <round> <name> <series> <round> <name>."""
    return len(score_tokens) == 6 and (
        is_number(score_tokens[0])
        and is_number(score_tokens[1])
        and not is_number(score_tokens[2])
        and is_number(score_tokens[3])
        and is_number(score_tokens[4])
        and not is_number(score_tokens[5])
    )


def check_format_10(score_tokens):
    """Tokens read as <any> <name> <series> <round> <name> <series> <round>."""
    return len(score_tokens) == 7 and (
        not is_number(score_tokens[1])
        and is_number(score_tokens[2])
        and is_number(score_tokens[3])
        and not is_number(score_tokens[4])
        and is_number(score_tokens[5])
        and is_number(score_tokens[6])
    )


def winner_of_round(p1_round, p2_round):
    """
    Determine if the round is 'won' by either player under typical table-tennis rules:
      - Must have 11 or more points,
      - Must be at least 2 points ahead.
    Return:
      1 if Player 1 wins the round,
      2 if Player 2 wins the round,
      None otherwise.
    """
    # If p1 >= 11 and leads by >=2, p1 wins
    if p1_round >= 11 and (p1_round - p2_round) >= 2:
        return 1
    # If p2 >= 11 and leads by >=2, p2 wins
    if p2_round >= 11 and (p2_round - p1_round) >= 2:
        return 2
    return None


def is_reset(score):
    """Checks if this scoreboard is a 'reset' (start of a new game)."""
    return score == (0, 0, 0, 0)


def is_valid_transition(prev_score, next_score):
    """
    Decide if it's a valid scoreboard transition from prev_score to next_score.

    For example, you might allow:
      1) The same score (no change).
      2) A 'reset' to (0, 0, 0, 0).
      3) Exactly one player's round score incremented by 1.
      4) A possible jump from something like (p1_series, 11, p2_series, x)
         to (p1_series+1, 0, p2_series, 0) for a new game, etc.

    You can adjust this logic based on your table‐tennis rules and data quirks.
    """
    if next_score == prev_score:
        return True

    # Always allow going to a reset
    if is_reset(next_score):
        return True

    # Example: If next_score is a "one‐point" increment from prev_score
    # This is simplistic: we only check that exactly one of p1_round or p2_round
    # went up by 1, with everything else staying the same. You can expand.
    p1s_a, p1r_a, p2s_a, p2r_a = prev_score
    p1s_b, p1r_b, p2s_b, p2r_b = next_score

    # same series, one round increments
    one_step_p1 = (p1s_b == p1s_a) and (p1r_b == p1r_a + 1) and (p2s_b == p2s_a) and (p2r_b == p2r_a)
    one_step_p2 = (p1s_b == p1s_a) and (p1r_b == p1r_a) and (p2s_b == p2s_a) and (p2r_b == p2r_a + 1)
    if one_step_p1 or one_step_p2:
        return True

    # Example: if a “game” might have ended, we might see the round scores reset
    # and one player’s “series” go up by 1. This is very approximate:
    new_game_p1 = p1s_b == p1s_a + 1 and p1r_b == 0 and p2s_b == p2s_a and p2r_b == 0
    new_game_p2 = p1s_b == p1s_a and p1r_b == 0 and p2s_b == p2s_a + 1 and p2r_b == 0
    return bool(new_game_p1 or new_game_p2)


def scores_from_raw_data(all_data):
    """Turn OCR records into [(frame, (series1, round1, series2, round2))] for the frames whose text parses as a scoreboard."""
    replacements = {"O": "0"}
    validation = []
    scores = []
    for ok, idx, result in all_data:
        if "OPTIM" in result:
            validation.append(2)
            continue
        if not ok:
            validation.append(0)
            continue

        result = result[0]
        if result is None:
            continue
        score_tokens = [r[1][0] for r in result]  # discards the bounding box and confidence measure
        for i, token in enumerate(score_tokens):
            if token in replacements:
                score_tokens[i] = replacements[token]
            if "." in token:
                score_tokens[i] = token.replace(".", "")

        if len(score_tokens) == 8:
            # use case: TOMOKAZU HARIMOTO 0 1 LIN SHIDONG 0 0 instead of TOMOKAZU/HARIMOTO 0 1 LIN/SHIDONG 0 0
            team_1 = score_tokens[0] + "/" + score_tokens[1]
            team_2 = score_tokens[4] + "/" + score_tokens[5]
            score_tokens = [team_1, score_tokens[2], score_tokens[3], team_2, score_tokens[6], score_tokens[7]]

        if check_format_1(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[1])
            first_round_score = int(score_tokens[2])
            second_series_score = int(score_tokens[4])
            second_round_score = int(score_tokens[5])
        elif check_format_2(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[1])
            first_round_score = int(score_tokens[2])
            second_series_score = int(score_tokens[3])
            second_round_score = int(score_tokens[4])
        elif check_format_3(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[0])
            first_round_score = int(score_tokens[1])
            second_series_score = int(score_tokens[4])
            second_round_score = int(score_tokens[5])
        elif check_format_4(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[2])
            first_round_score = int(score_tokens[3])
            second_series_score = int(score_tokens[5])
            second_round_score = int(score_tokens[6])
        elif check_format_5(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[1])
            first_round_score = int(score_tokens[2])
            second_series_score = int(score_tokens[5])
            second_round_score = int(score_tokens[6])
        elif check_format_6(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[1])
            first_round_score = int(score_tokens[2])
            second_series_score = int(score_tokens[4])
            second_round_score = int(score_tokens[5])
        elif check_format_7(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[3])
            first_round_score = int(score_tokens[4])
            second_series_score = int(score_tokens[7])
            second_round_score = int(score_tokens[8])
        elif check_format_8(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[2])
            first_round_score = int(score_tokens[3])
            second_series_score = int(score_tokens[4])
            second_round_score = int(score_tokens[5])
        elif check_format_9(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[0])
            first_round_score = int(score_tokens[1])
            second_series_score = int(score_tokens[3])
            second_round_score = int(score_tokens[4])
        elif check_format_10(score_tokens):
            validation.append(1)
            first_series_score = int(score_tokens[2])
            first_round_score = int(score_tokens[3])
            second_series_score = int(score_tokens[5])
            second_round_score = int(score_tokens[6])
        else:
            validation.append(-1)
            continue
        scores.append((idx, (first_series_score, first_round_score, second_series_score, second_round_score)))
    return scores


def changes_from_scores(scores):
    """Walk the score sequence and return [((start frame, end frame), score before, score after)] for every valid score change."""
    current_start_idx, current_score = scores[0]

    changes = []
    for idx, score in scores:
        # a proper score increase; otherwise wait until the correct next score shows up
        if any(score[i] != current_score[i] for i in range(4)) and is_valid_transition(current_score, score):
            changes.append(((current_start_idx, idx), current_score, score))
            current_start_idx, current_score = idx, score

    return changes
