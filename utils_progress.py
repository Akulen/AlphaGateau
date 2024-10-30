from collections import deque
from dataclasses import dataclass
from math import e, ceil
from rich.table import Column
from rich.text import Text
from typing import Any, Optional
import rich.progress as rp


def resume_task(progress: rp.Progress, task_id: rp.TaskID) -> None:
    with progress._lock:
        task = progress._tasks[task_id]
        if task.start_time is None:
            progress.start_task(task_id)
        elif task.stop_time is not None:
            current_time = progress.get_time()
            delta = current_time - task.stop_time
            def foo(x):
                return x._replace(timestamp=x.timestamp + delta)
            task._progress = deque(map(foo, task._progress))
            task.start_time = task.start_time + delta
            task.stop_time = None

class ProgressEMA(rp.Progress):
    def update(
        self,
        task_id: rp.TaskID,
        *,
        total: Optional[float] = None,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        visible: Optional[bool] = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        """Update information associated with a task.

        Args:
            task_id (TaskID): Task id (returned by add_task).
            total (float, optional): Updates task.total if not None.
            completed (float, optional): Updates task.completed if not None.
            advance (float, optional): Add a value to task.completed if not None.
            description (str, optional): Change task description if not None.
            visible (bool, optional): Set visible flag if not None.
            refresh (bool): Force a refresh of progress information. Default is False.
            **fields (Any): Additional data fields required for rendering.
        """
        with self._lock:
            task = self._tasks[task_id]
            completed_start = task.completed

            if total is not None and total != task.total:
                task.total = total
                task._reset()
            if advance is not None:
                task.completed += advance
            if completed is not None:
                task.completed = completed
            if description is not None:
                task.description = description
            if visible is not None:
                task.visible = visible
            task.fields.update(fields)
            update_completed = task.completed - completed_start

            current_time = self.get_time()
            old_sample_time = current_time - self.speed_estimate_period
            _progress = task._progress

            popleft = _progress.popleft
            while _progress and _progress[0].timestamp < old_sample_time:
                popleft()
            if update_completed > 0:
                _progress.append(rp.ProgressSample(current_time, update_completed))
            if (
                task.total is not None
                and task.completed >= task.total
                and task.finished_time is None
            ):
                task.finished_time = task.elapsed

        if refresh:
            self.refresh()

    def advance(self, task_id: rp.TaskID, advance: float = 1) -> None:
        """Advance task by a number of steps.

        Args:
            task_id (TaskID): ID of task.
            advance (float): Number of steps to advance. Default is 1.
        """
        current_time = self.get_time()
        with self._lock:
            task = self._tasks[task_id]
            completed_start = task.completed
            task.completed += advance
            update_completed = task.completed - completed_start
            old_sample_time = current_time - self.speed_estimate_period
            _progress = task._progress

            popleft = _progress.popleft
            while _progress and _progress[0].timestamp < old_sample_time:
                popleft()
            while len(_progress) > 1000:
                popleft()
            _progress.append(rp.ProgressSample(current_time, update_completed))
            if (
                task.total is not None
                and task.completed >= task.total
                and task.finished_time is None
            ):
                task.finished_time = task.elapsed
                task.finished_speed = task.speed


class TimeRemainingColumn(rp.ProgressColumn):
    """Renders estimated time remaining.

    Args:
        compact (bool, optional): Render MM:SS when time remaining is less than an hour. Defaults to False.
        elapsed_when_finished (bool, optional): Render time elapsed when the task is finished. Defaults to False.
        exponential_moving_average (bool, optional): Estimate using an exponential moving average instead of averaging the time over a past window. Defaults to False.
    """

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(
        self,
        compact: bool = False,
        elapsed_when_finished: bool = False,
        exponential_moving_average: bool = False,
        table_column: Optional[Column] = None,
    ):
        self.compact = compact
        self.elapsed_when_finished = elapsed_when_finished
        self.exponential_moving_average = exponential_moving_average
        super().__init__(table_column=table_column)

    def render(self, task: "Task") -> Text: #type: ignore
        """Show time remaining."""
        if self.elapsed_when_finished and task.finished:
            task_time = task.finished_time
            style = "progress.elapsed"
        else:
            if self.exponential_moving_average:
                task_time = task.time_remaining_ema
            else:
                task_time = task.time_remaining
            style = "progress.remaining"

        if task.total is None:
            return Text("", style=style)

        if task_time is None:
            return Text("--:--" if self.compact else "-:--:--", style=style)

        # Based on https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
        minutes, seconds = divmod(int(task_time), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        return Text(formatted, style=style)


@dataclass
class TaskEMA(rp.Task):
    speed_ema: Optional[float] = None
    """Optional[float]: The current speed estimated using an exponential moving average."""

    def update_speed_ema(
        self, update_completed: float, update_time: float, alpha: float = 0.1
    ) -> None:
        update_speed = update_completed / update_time
        if self.speed_ema is None:
            self.speed_ema = update_speed
        else:
            weight_old_speed = (1 - alpha) ** update_completed
            self.speed_ema = 1 / (
                (1 - weight_old_speed) / update_speed
                + weight_old_speed / self.speed_ema
            )  # Harmonic mean of the speeds

    @property
    def time_remaining_ema(self) -> Optional[float]:
        """Optional[float]: Get estimated time to completion using an exponential moving average, or ``None`` if no data."""
        if self.finished:
            return 0.0
        speed = self.speed_ema
        if not speed:
            return None
        remaining = self.remaining
        if remaining is None:
            return None
        estimate = remaining / speed
        current_step_progress = 0.0
        if self.stop_time is None:
            with self._lock:
                progress = self._progress
                if progress:
                    current_step_progress = self.get_time() - progress[-1].timestamp
                elif self.start_time is not None:
                    current_step_progress = self.get_time() - self.start_time
            current_step_progress = (
                1 / speed * (1 - e ** (-current_step_progress * speed))
            )
        return ceil(estimate - current_step_progress)

    def _reset(self) -> None:
        """Reset progress."""
        super()._reset()
        self.speed_ema = None


class Progress2(rp.Progress):
    def __init__(
        self,
        *args: Any,
        speed_estimate_alpha: float = 0.1,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.speed_estimate_alpha = speed_estimate_alpha

    def update(
        self,
        task_id: rp.TaskID,
        *,
        total: Optional[float] = None,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        visible: Optional[bool] = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        """Update information associated with a task.

        Args:
            task_id (TaskID): Task id (returned by add_task).
            total (float, optional): Updates task.total if not None.
            completed (float, optional): Updates task.completed if not None.
            advance (float, optional): Add a value to task.completed if not None.
            description (str, optional): Change task description if not None.
            visible (bool, optional): Set visible flag if not None.
            refresh (bool): Force a refresh of progress information. Default is False.
            **fields (Any): Additional data fields required for rendering.
        """
        with self._lock:
            task = self._tasks[task_id]
            completed_start = task.completed

            if total is not None and total != task.total:
                task.total = total
                task._reset()
            if advance is not None:
                task.completed += advance
            if completed is not None:
                task.completed = completed
            if description is not None:
                task.description = description
            if visible is not None:
                task.visible = visible
            task.fields.update(fields)
            update_completed = task.completed - completed_start

            current_time = self.get_time()
            old_sample_time = current_time - self.speed_estimate_period
            _progress = task._progress
            update_time = None
            if _progress:
                update_time = current_time - _progress[-1].timestamp
            elif task.start_time is not None:
                update_time = current_time - task.start_time

            popleft = _progress.popleft
            while len(_progress) > 2 and _progress[0].timestamp < old_sample_time:
                popleft()
            if update_completed > 0:
                _progress.append(rp.ProgressSample(current_time, update_completed))
            if update_completed > 0 and update_time is not None and update_time > 0:
                task.update_speed_ema(
                    update_completed,
                    update_time,
                    max(self.speed_estimate_alpha, 1 / (1 + completed_start)),
                )
            if (
                task.total is not None
                and task.completed >= task.total
                and task.finished_time is None
            ):
                task.finished_time = task.elapsed

        if refresh:
            self.refresh()

