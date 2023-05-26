import gin
import numpy as np


def process_multihorizon_task_string(task):
    """Simple wrapper to get multi horizon label names"""
    if '-' in task:
        import re
        task_list = []
        start, stop, step = re.findall(r'\d+', task)
        to_replace = '-'.join([start, stop, step])
        for t in np.arange(*tuple(map(int, (start, stop, step)))):
            task_list.append(task.replace(to_replace, str(t)))
        task_list.append(task.replace(to_replace, stop))
        task = task_list
    else:
        task = [task]
    return task


@gin.configurable()
def q_exp_param(dt, h_true, h_min, h_max, delta_h=1, gamma=0.01):
    """ Exponential smoothing function.

    Args:
        dt: (int) distance steps to next event (+inf if no event).
        h_true: (int) true horizon of prediction in steps. (not used)
        h_min: (int) minimal horizon of smoothing in steps.
        h_max: (int) maximum horizon of smoothing in steps.
        delta_h: (float) number of step per hour.
        gamma: (float) positive smoothing strength parameter.

    Returns:
        q^exp(1|t) as a float
    """
    if dt <= h_min:
        return 1
    elif dt > h_max:
        return 0
    else:
        h_min_scaled = h_min / delta_h
        h_max_scaled = h_max / delta_h
        dt_scaled = dt / delta_h

        d = -(1 / gamma) * np.log(np.exp(-gamma * (h_min_scaled)) - np.exp(-gamma * (h_max_scaled)))
        A = -np.exp(-gamma * (h_max_scaled - d))
        return np.exp(-gamma * (dt_scaled - d)) + A


@gin.configurable()
def q_sigmoid_param(dt, h_true, h_min, h_max, delta_h=1, gamma=0.1):
    """ Sigmoidal smoothing function.

    Args:
        dt: (int) distance steps to next event (+inf if no event).
        h_true: (int) true horizon of prediction in steps.
        h_min: (int) minimal horizon of smoothing in steps.
        h_max: (int) maximum horizon of smoothing in steps.
        delta_h: (float) number of step per hour.
        gamma: (float) positive smoothing strength parameter.

    Returns:
        q^sigmoid(1|t) as a float
    """
    if dt <= h_min:
        return 1
    elif dt > h_max:
        return 0
    else:
        beta = 1 / gamma
        h_min_scaled = h_min / delta_h
        h_max_scaled = h_max / delta_h
        h_true_scaled = h_true / delta_h

        dt_scaled = dt / delta_h

        norm = np.exp(beta * (h_max_scaled))
        if norm == np.inf:  # Overflow
            return q_kro(dt, h_true, h_min, h_max)

        # In the general case d != h and depends on h_min and h_max in the following way.
        D_1 = np.exp(beta * (h_min_scaled)) - np.exp(beta * (h_max_scaled))
        D_2 = np.exp(beta * (h_true_scaled)) - np.exp(beta * (h_max_scaled))
        D_1 /= np.exp(beta * (h_max_scaled))  # Scale for overflow
        D_2 /= np.exp(beta * (h_max_scaled))  # Scale for overflow
        n = - D_1 / 2 + D_2
        m = D_1 * np.exp(beta * (h_true_scaled)) / 2 - D_2 * np.exp(beta * (h_min_scaled))
        Q = n / m
        d = -(1 / beta) * np.log(Q)

        # Lower asymptote
        A = (np.exp(beta * (h_min_scaled - d)) + 1) / (
                (np.exp(beta * (h_min_scaled - d)) + 1) - (np.exp(beta * (h_max_scaled - d)) + 1))

        # Capacity
        K = -A * np.exp(beta * (h_max_scaled - d))

        return (K - A) / (1 + np.exp(beta * (dt_scaled - d))) + A


@gin.configurable()
def q_step(dt, h_true, h_min, h_max, delta_h=12, gamma=2):
    """ Step smoothing function.

    Args:
        dt: (int) distance steps to next event (+inf if no event).
        h_true: (int) true horizon of prediction in steps. (not used)
        h_min: (int) minimal horizon of smoothing in steps.
        h_max: (int) maximum horizon of smoothing in steps.
        delta_h: (float) number of step per hour.
        gamma: (float) step size in hours, thus 1/delta_h is linear.

    Returns:
        q^step(1|t) as a float
    """
    if dt < h_min:
        return 1
    elif dt > h_max:
        return 0
    else:
        return (np.ceil((h_max - dt) / (gamma * delta_h))) / (((h_max - h_min) // (gamma * delta_h)))


@gin.configurable()
def q_linear(dt, h_true, h_min, h_max, delta_h=12, gamma=0.1):
    """ Step smoothing function.

    Args:
        dt: (int) distance steps to next event (+inf if no event).
        h_true: (int) true horizon of prediction in steps. (not used)
        h_min: (int) minimal horizon of smoothing in steps.
        h_max: (int) maximum horizon of smoothing in steps.
        delta_h: (float) number of step per hour. (not used)
        gamma: (not used)

    Returns:
        q^linear(1|t) as a float
    """
    if dt <= h_min:
        return 1
    elif dt > h_max:
        return 0
    else:
        return (h_max - dt) / (h_max - h_min + 1e-8)


@gin.configurable()
def q_kro(dt, h_true, h_min, h_max, delta_h=12, gamma=0.1):
    if dt < h_true:
        return 1
    else:
        return 0


@gin.configurable()
def q_ls(dt, h_true, h_min, h_max, delta_h=12, gamma=0.1):
    if dt <= h_true:
        return 1 - gamma
    elif dt > h_true:
        return gamma
    else:
        return -1


@gin.configurable('get_smoothed_labels')
def get_smoothed_labels(label, event, smoothing_fn=gin.REQUIRED, h_true=gin.REQUIRED, h_min=gin.REQUIRED,
                        h_max=gin.REQUIRED, delta_h=12, gamma=0.1):
    diffs = np.concatenate([np.zeros(1), event[1:] - event[:-1]], axis=-1)
    pos_event_change_full = np.where((diffs == 1) & (event == 1))[0]

    multihorizon = isinstance(h_true, list)
    if multihorizon:
        label_for_event = label[0]
        h_for_event = h_true[0]
    else:
        label_for_event = label
        h_for_event = h_true
    diffs_label = np.concatenate([np.zeros(1), label_for_event[1:] - label_for_event[:-1]], axis=-1)

    # Event that occurred after the end of the stay for M3B.
    # In that case event are equal to the number of hours after the end of stay when the event occured.
    pos_event_change_delayed = np.where((diffs >= 1) & (event > 1))[0]
    if len(pos_event_change_delayed) > 0:
        delays = event[pos_event_change_delayed] - 1
        pos_event_change_delayed += delays.astype(int)
        pos_event_change_full = np.sort(np.concatenate([pos_event_change_full, pos_event_change_delayed]))

    last_know_label = label_for_event[np.where(label_for_event != -1)][-1]
    last_know_idx = np.where(label_for_event == last_know_label)[0][-1]

    # Need to handle the case where the ts was truncatenated at 2016 for HiB
    if ((last_know_label == 1) and (len(pos_event_change_full) == 0)) or (
            (last_know_label == 1) and (last_know_idx >= pos_event_change_full[-1])):
        last_know_event = 0
        if len(pos_event_change_full) > 0:
            last_know_event = pos_event_change_full[-1]

        last_known_stable = 0
        known_stable = np.where(label_for_event == 0)[0]
        if len(known_stable) > 0:
            last_known_stable = known_stable[-1]

        pos_change = np.where((diffs_label >= 1) & (label_for_event == 1))[0]
        last_pos_change = pos_change[np.where(pos_change > max(last_know_event, last_known_stable))][0]
        pos_event_change_full = np.concatenate([pos_event_change_full, [last_pos_change + h_for_event]])

    # No event case
    if len(pos_event_change_full) == 0:
        pos_event_change_full = np.array([np.inf])

    time_array = np.arange(len(label))
    dist = pos_event_change_full.reshape(-1, 1) - time_array
    dte = np.where(dist > 0, dist, np.inf).min(axis=0)
    if multihorizon:
        smoothed_labels = []
        for k in range(label.shape[-1]):
            smoothed_labels.append(np.array(list(
                map(lambda x: smoothing_fn(x, h_true=h_true[k], h_min=h_min[k], h_max=h_max[k], delta_h=delta_h,
                                           gamma=gamma), dte))))
        return np.stack(smoothed_labels, axis=-1)
    else:
        return np.array(
            list(map(lambda x: smoothing_fn(x, h_true=h_true, h_min=h_min,
                                            h_max=h_max, delta_h=delta_h, gamma=gamma), dte)))
