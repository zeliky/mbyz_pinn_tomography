"""Training logging: standard logging (console + file) and optional HTML reports with plots."""

from __future__ import annotations

import base64
import logging
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAINING_LOGGER_NAME = "tomo.training"


def setup_training_logging(
    log_dir: Path | str,
    *,
    run_id: str | None = None,
    logger_name: str = TRAINING_LOGGER_NAME,
) -> tuple[logging.Logger, str]:
    """Attach file + stdout handlers under *log_dir*; return logger and *run_id*."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    rid = run_id or datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    (log_path / "formated_datetime.txt").write_text(rid + "\n", encoding="utf-8")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path / f"terminal_{rid}.txt", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, rid


def _insert_before_body_close(lines: list[str], fragments: list[str]) -> None:
    insert_index: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if "</body>" in lines[i]:
            insert_index = i
            break
    if insert_index is None:
        raise ValueError("No </body> tag found in HTML file.")
    for frag in reversed(fragments):
        lines.insert(insert_index, frag)


class HtmlTrainingReport:
    """Append-only HTML log with text and base64 PNG figures (before </body>)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "<html>\n<body>\n</body>\n</html>\n",
                encoding="utf-8",
            )

    def add_text(self, *parts: Any) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        fragments: list[str] = []
        for p in parts:
            fragments.append("<br>\n")
            fragments.append(str(p) + "\n")
        _insert_before_body_close(lines, fragments)
        self.path.write_text("".join(lines), encoding="utf-8")

    def add_figure(self, fig: plt.Figure) -> None:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig)

        image_html = f'<img src="data:image/png;base64,{b64}" alt="Plot Image">\n'
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        _insert_before_body_close(lines, [image_html])
        self.path.write_text("".join(lines), encoding="utf-8")


class EpochMetricsLogger:
    """Log epoch lines and append loss / MSE curves to HTML when validation runs."""

    def __init__(
        self,
        logger: logging.Logger,
        report: HtmlTrainingReport | None,
        *,
        total_epochs: int,
    ) -> None:
        self._logger = logger
        self._report = report
        self._total_epochs = total_epochs
        self._epochs: list[int] = []
        self._train_loss: list[float] = []
        self._train_mse: list[float] = []
        self._val_loss: list[float | None] = []
        self._val_mse: list[float | None] = []

    def on_epoch_end(
        self,
        epoch_1based: int,
        train_loss: float,
        train_mse_recon: float,
        *,
        val_loss: float | None = None,
        val_mse_recon: float | None = None,
    ) -> None:
        if val_loss is not None and val_mse_recon is not None:
            msg = (
                f"epoch {epoch_1based}/{self._total_epochs}  train_loss={train_loss:.4f}  "
                f"train_mse_recon={train_mse_recon:.6f}  "
                f"val_loss={val_loss:.4f}  val_mse_recon={val_mse_recon:.6f}"
            )
        else:
            msg = (
                f"epoch {epoch_1based}/{self._total_epochs}  train_loss={train_loss:.4f}  "
                f"train_mse_recon={train_mse_recon:.6f}"
            )
        self._logger.info(msg)
        if self._report is not None:
            self._report.add_text(msg)

        self._epochs.append(epoch_1based)
        self._train_loss.append(train_loss)
        self._train_mse.append(train_mse_recon)
        self._val_loss.append(val_loss)
        self._val_mse.append(val_mse_recon)

        if self._report is not None and val_loss is not None and val_mse_recon is not None:
            self._append_curves_figure()

    def _append_curves_figure(self) -> None:
        assert self._report is not None
        ep = self._epochs
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(ep, self._train_loss, label="train_loss", color="C0")
        v_ep = [e for e, v in zip(ep, self._val_loss, strict=True) if v is not None]
        v_loss = [v for v in self._val_loss if v is not None]
        if v_ep:
            axes[0].plot(v_ep, v_loss, label="val_loss", color="C1")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ep, self._train_mse, label="train_mse_recon", color="C0")
        v_mse = [v for v in self._val_mse if v is not None]
        if v_ep and len(v_mse) == len(v_ep):
            axes[1].plot(v_ep, v_mse, label="val_mse_recon", color="C1")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("MSE recon")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("Stage 0 training curves")
        fig.tight_layout()
        self._report.add_figure(fig)


def log_stage2_episode_series(
    logger: logging.Logger,
    report: HtmlTrainingReport | None,
    episode_infos: list[dict[str, Any]],
    *,
    title: str = "Stage 2 episode reward",
) -> None:
    """Log aggregate stats and optional HTML plot of reward (and steps) per episode."""
    n = len(episode_infos)
    if n == 0:
        logger.info("Stage 2: no episodes completed.")
        return

    rewards = [float(e.get("reward", 0.0)) for e in episode_infos]
    steps = [int(e.get("steps", 0)) for e in episode_infos]
    mean_r = sum(rewards) / n
    logger.info("Stage 2 training done. episodes=%d mean_reward=%.6f", n, mean_r)
    if report is not None:
        report.add_text(f"Stage 2: {n} episodes, mean_reward={mean_r:.6f}")

    if report is None:
        return

    x = list(range(n))
    fig, ax_r = plt.subplots(figsize=(8, 4))
    ax_r.plot(x, rewards, color="C0", label="reward")
    ax_r.set_xlabel("episode")
    ax_r.set_ylabel("reward")
    ax_r.grid(True, alpha=0.3)

    ax_s = ax_r.twinx()
    ax_s.plot(x, steps, color="C1", alpha=0.7, label="steps")
    ax_s.set_ylabel("steps")
    fig.suptitle(title)
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    lines_s, labels_s = ax_s.get_legend_handles_labels()
    ax_r.legend(lines_r + lines_s, labels_r + labels_s, loc="upper right")
    fig.tight_layout()
    report.add_figure(fig)
