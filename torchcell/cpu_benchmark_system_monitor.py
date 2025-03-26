import concurrent.futures
import time
import numpy as np
import multiprocessing as mp
import os
import signal
import argparse
import psutil
import threading
import queue
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""

    title: str
    matrix_size: int
    num_cpus: int
    repeats: int
    base_seed: int
    output_dir: Path

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.matrix_size <= 0:
            raise ValueError("Matrix size must be positive")
        if self.num_cpus <= 0:
            raise ValueError("Number of CPUs must be positive")
        if self.repeats <= 0:
            raise ValueError("Number of repeats must be positive")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


class IPMICollector:
    """Handles IPMI data collection and parsing"""

    def __init__(self, timeout: int = 30, retry_count: int = 3):
        self.timeout = timeout
        self.retry_count = retry_count

    def collect_data(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Collect IPMI data with retries"""
        for attempt in range(self.retry_count):
            try:
                result = subprocess.run(
                    "ipmitool sensor list | grep -E 'Temp\\.|RPM'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode == 0:
                    return self.parse_output(result.stdout)
                print(
                    f"IPMI command failed (attempt {attempt + 1}/{self.retry_count}): {result.stderr}"
                )
            except subprocess.TimeoutExpired:
                print(
                    f"IPMI command timed out after {self.timeout} seconds (attempt {attempt + 1}/{self.retry_count})"
                )
            except subprocess.SubprocessError as e:
                print(
                    f"IPMI command error (attempt {attempt + 1}/{self.retry_count}): {e}"
                )

            if attempt < self.retry_count - 1:
                time.sleep(1)

        return {}, {}

    @staticmethod
    def parse_output(output: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Parse IPMI output into temperature and RPM data"""
        temps = {}
        rpms = {}

        for line in output.split("\n"):
            if not line or "Sensor Name" in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue

            name = parts[0].strip()
            try:
                value = float(parts[1]) if parts[1].strip() != "na" else np.nan
            except ValueError:
                continue

            if "Temp." in name:
                temps[name] = value
            elif "RPM" in parts[2]:
                rpms[name] = value

        return temps, rpms


class SystemMonitor:
    """Monitors system metrics during benchmark runs"""

    def __init__(self, config: BenchmarkConfig, data_queue: queue.Queue):
        self.config = config
        self.data_queue = data_queue
        self.temperatures: List[Dict[str, float]] = []
        self.rpms: List[Dict[str, float]] = []
        self.timestamps: List[datetime] = []
        self.benchmark_points: List[int] = []
        self.benchmark_times: List[float] = []
        self.run_numbers: List[int] = []
        self.current_run = 1
        self.ipmi_collector = IPMICollector()
        self.stop_event = threading.Event()

    def _validate_data(self) -> bool:
        """Validate collected data"""
        if not self.timestamps:
            return False
        if len(self.timestamps) != len(self.temperatures):
            return False
        if len(self.timestamps) != len(self.rpms):
            return False
        if len(self.timestamps) != len(self.run_numbers):
            return False
        return True

    def collect_data(self) -> None:
        """Collect system metrics data"""
        while not self.stop_event.is_set():
            try:
                # Check for benchmark completion signal
                try:
                    benchmark_time = self.data_queue.get_nowait()
                    self.benchmark_points.append(len(self.timestamps))
                    self.benchmark_times.append(benchmark_time)
                    self.current_run += 1
                    print(f"Starting run {self.current_run}")
                except queue.Empty:
                    pass

                # Collect IPMI data
                temps, rpms = self.ipmi_collector.collect_data()
                if not temps and not rpms:
                    print("Warning: No IPMI data collected in this iteration")
                    continue

                self.temperatures.append(temps)
                self.rpms.append(rpms)
                self.timestamps.append(datetime.now())
                self.run_numbers.append(self.current_run)

                time.sleep(2)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in data collection: {e}")

    def export_data(self) -> None:
        """Export collected data to CSV"""
        if not self._validate_data():
            print("Warning: Data validation failed, skipping export")
            return

        data = {
            "timestamp": self.timestamps,
            "run_number": self.run_numbers,
            **{
                f"temp_{k}": [d.get(k, np.nan) for d in self.temperatures]
                for k in set().union(*self.temperatures)
            },
            **{
                f"rpm_{k}": [d.get(k, np.nan) for d in self.rpms]
                for k in set().union(*self.rpms)
            },
        }

        df = pd.DataFrame(data)
        output_path = (
            self.config.output_dir
            / f"{self.config.title}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(output_path, index=False)
        print(f"Data exported to: {output_path}")

    def plot_data(self) -> Optional[Path]:
        """Create visualization of collected data with continuous timeline"""
        if not self._validate_data():
            print("Warning: No valid monitoring data available for plotting")
            return None

        plt.style.use("torchcell/torchcell.mplstyle")

        # Create figure with shared x-axis - increased figure size
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 12))

        # Add more spacing at the right for legends
        plt.subplots_adjust(right=0.85)

        fig.suptitle(
            f"System Monitoring - {self.config.title}\n"
            + f"Benchmark Stats: Mean={np.mean(self.benchmark_times):.2f}s, "
            + f"Std={np.std(self.benchmark_times):.2f}s"
        )

        # Define markers to cycle through
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "8", "H"]
        marker_idx = 0

        # Get colors from style
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_idx = 0

        # Create mappings for each device
        device_styles = {}

        # Map colors and markers to temperature sensors
        for sensor in ["CPU Temp.", "LAN Temp."]:
            device_styles[sensor] = {
                "color": colors[color_idx],
                "marker": markers[marker_idx],
            }
            color_idx = (color_idx + 1) % len(colors)
            marker_idx = (marker_idx + 1) % len(markers)

        for x in "ABCDEFGH":
            sensor = f"DIMM{x}1 Temp."
            device_styles[sensor] = {
                "color": colors[color_idx],
                "marker": markers[marker_idx],
            }
            color_idx = (color_idx + 1) % len(colors)
            marker_idx = (marker_idx + 1) % len(markers)

        for i in range(1, 8):
            sensor = f"PCIE0{i} Temp."
            device_styles[sensor] = {
                "color": colors[color_idx],
                "marker": markers[marker_idx],
            }
            color_idx = (color_idx + 1) % len(colors)
            marker_idx = (marker_idx + 1) % len(markers)

        for fan in [
            "CPU_FAN",
            "CPU_OPT",
            "CHA_FAN2",
            "CHA_FAN3",
            "CHA_FAN4",
            "CHA_FAN6",
            "SOC_FAN",
        ]:
            device_styles[fan] = {
                "color": colors[color_idx],
                "marker": markers[marker_idx],
            }
            color_idx = (color_idx + 1) % len(colors)
            marker_idx = (marker_idx + 1) % len(markers)

        # Process data
        temp_df = pd.DataFrame(self.temperatures)
        rpm_df = pd.DataFrame(self.rpms)

        # Calculate continuous timeline
        start_time = self.timestamps[0]
        x_values = [(t - start_time).total_seconds() for t in self.timestamps]

        # Store handles for legends
        temp_handles = []
        temp_labels = []
        rpm_handles = []
        rpm_labels = []

        # Plot temperatures
        for sensor in temp_df.columns:
            if any(not np.isnan(y) for y in temp_df[sensor].values):
                group_name = (
                    "CPU"
                    if "CPU" in sensor
                    else (
                        "Memory"
                        if "DIMM" in sensor
                        else (
                            "PCIe"
                            if "PCIE" in sensor
                            else "Network" if "LAN" in sensor else "Other"
                        )
                    )
                )

                style = device_styles[sensor]
                line = ax1.plot(
                    x_values,
                    temp_df[sensor].values,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=4,
                    markevery=5,  # Show marker every 5 points to avoid crowding
                    alpha=0.8,
                    label=f"{group_name}: {sensor}",
                )
                temp_handles.append(line[0])
                temp_labels.append(f"{group_name}: {sensor}")

        # Plot fan speeds
        for fan in rpm_df.columns:
            if any(not np.isnan(y) for y in rpm_df[fan].values):
                group_name = (
                    "CPU Fans"
                    if "CPU" in fan
                    else "Chassis Fans" if "CHA" in fan else "Other"
                )

                style = device_styles[fan]
                line = ax2.plot(
                    x_values,
                    rpm_df[fan].values,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=4,
                    markevery=5,  # Show marker every 5 points to avoid crowding
                    alpha=0.8,
                    label=f"{group_name}: {fan}",
                )
                rpm_handles.append(line[0])
                rpm_labels.append(f"{group_name}: {fan}")

        # Add completion markers
        cumulative_time = 0
        for i, completion_time in enumerate(self.benchmark_times, 1):
            cumulative_time += completion_time
            for ax in [ax1, ax2]:
                ax.axvline(x=cumulative_time, color="r", linestyle="--", alpha=0.3)
                ax.text(
                    cumulative_time,
                    ax.get_ylim()[1],
                    f"Run {i}: {completion_time:.1f}s",
                    rotation=90,
                    verticalalignment="top",
                    fontsize=8,
                    alpha=0.7,
                )

        # Add legends with adjusted positioning
        # Temperature sensors legend
        ax1.legend(
            temp_handles,
            temp_labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            title="Temperature Sensors",
        )

        # Fan speeds legend
        ax2.legend(
            rpm_handles,
            rpm_labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            title="Fan Speeds",
        )

        # Customize plots
        ax1.set_title(
            f"Configuration: Matrix Size={self.config.matrix_size}, CPUs={self.config.num_cpus}"
        )
        ax1.set_ylabel("Temperature (°C)")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Fan Speed (RPM)")

        # Set fine-grained temperature y-axis ticks
        # Get the current y limits
        ymin, ymax = ax1.get_ylim()
        # Create ticks every 2°C
        ax1.yaxis.set_major_locator(plt.MultipleLocator(2))
        # Add minor ticks every 1°C
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
        # Adjust grid for readability
        ax1.grid(True, which="major", alpha=0.5)
        ax1.grid(True, which="minor", alpha=0.2)

        # Adjust temperature axis label size and rotation
        ax1.tick_params(axis="y", labelsize=8, rotation=0)

        # Adjust fan speed axis for better readability
        ax2.yaxis.set_major_locator(plt.MultipleLocator(50))
        ax2.tick_params(axis="y", labelsize=8)

        plt.tight_layout()

        # Save plot with increased DPI for better quality
        output_path = (
            self.config.output_dir
            / f"{self.config.title}_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to: {output_path}")
        plt.close()

        return output_path

    def _plot_data_series(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        groups: Dict[str, List[str]],
        colors: Dict[str, str],
        markers: List[str],
        legend_title: str,
    ) -> None:
        """Helper method to plot data series with consistent styling"""
        for run_num in range(1, self.current_run):
            run_mask = [r == run_num for r in self.run_numbers]
            run_indices = [i for i, mask in enumerate(run_mask) if mask]

            if not run_indices:
                continue

            run_start = self.timestamps[run_indices[0]]
            x_values = [
                (t - run_start).total_seconds()
                for t, mask in zip(self.timestamps, run_mask)
                if mask
            ]

            for group_name, sensors in groups.items():
                for sensor in sensors:
                    if sensor in df.columns:
                        y_values = [y for y, mask in zip(df[sensor], run_mask) if mask]
                        if any(not np.isnan(y) for y in y_values):
                            ax.plot(
                                x_values,
                                y_values,
                                color=colors[group_name],
                                alpha=0.7,
                                marker=markers[run_num - 1],
                                markersize=4,
                                label=(
                                    f"{group_name}: {sensor}"
                                    if run_num == 1
                                    else "_nolegend_"
                                ),
                            )

            if run_num <= len(self.benchmark_times):
                self._add_completion_marker(ax, self.benchmark_times[run_num - 1])

        ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", title=legend_title)

    def _add_completion_marker(self, ax: plt.Axes, completion_time: float) -> None:
        """Add completion marker to plot"""
        ax.axvline(x=completion_time, color="r", linestyle="--", alpha=0.3)
        ax.text(
            completion_time,
            ax.get_ylim()[1],
            f"{completion_time:.1f}s",
            rotation=90,
            verticalalignment="top",
            fontsize=8,
            alpha=0.7,
        )


class Benchmark:
    """Handles benchmark execution and monitoring"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.monitor: Optional[SystemMonitor] = None
        self.monitor_queue: Optional[queue.Queue] = None

    def setup_monitoring(self) -> None:
        """Initialize system monitoring"""
        self.monitor_queue = queue.Queue()
        self.monitor = SystemMonitor(self.config, self.monitor_queue)

        monitor_thread = threading.Thread(
            target=self._run_monitor_on_last_cpu, args=(self.monitor,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()

    @staticmethod
    def _run_monitor_on_last_cpu(monitor: SystemMonitor) -> None:
        """Run monitor on last CPU"""
        try:
            total_cpus = psutil.cpu_count()
            p = psutil.Process()
            p.cpu_affinity([total_cpus - 1])
            monitor.collect_data()
        except Exception as e:
            print(f"Failed to set monitoring CPU affinity: {e}")
            monitor.collect_data()

    def run(self) -> List[float]:
        """Run the complete benchmark suite"""
        times = []
        seeds = [
            self.config.base_seed + i * self.config.num_cpus
            for i in range(self.config.repeats)
        ]

        for i, seed in enumerate(seeds, 1):
            print(f"Running benchmark {i}/{self.config.repeats}")
            duration = self._run_single_benchmark(seed)
            print(f"Duration: {duration:.2f}s")
            times.append(duration)
            if self.monitor_queue:
                self.monitor_queue.put(duration)

        return times

    def _run_single_benchmark(self, seed: int) -> float:
        """Run a single benchmark iteration"""
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.config.num_cpus, initializer=self._init_worker
        ) as executor:
            start_time = time.time()
            list(
                executor.map(
                    matrix_multiply,
                    [
                        (self.config.matrix_size, s)
                        for s in range(seed, seed + self.config.num_cpus)
                    ],
                )
            )
            return time.time() - start_time

    @staticmethod
    def _init_worker() -> None:
        """Initialize worker process"""
        try:
            total_cpus = psutil.cpu_count()
            p = psutil.Process()
            p.cpu_affinity(list(range(total_cpus - 1)))
        except Exception as e:
            print(f"Failed to set worker CPU affinity: {e}")


def matrix_multiply(size_seed: tuple) -> None:
    """Matrix multiplication benchmark function"""
    size, seed = size_seed
    np.random.seed(seed)
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    np.dot(A, B)


def main() -> None:
    """Main program entry point"""
    parser = argparse.ArgumentParser(description="CPU Benchmark with System Monitoring")
    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help='Title for the monitoring session (e.g., "2-DIMMs")',
    )
    parser.add_argument(
        "--repeats", type=int, default=2, help="Number of benchmark repeats"
    )
    parser.add_argument(
        "--matrix-size", type=int, default=4000, help="Size of matrices to multiply"
    )
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory for storing results",
    )

    args = parser.parse_args()

    # Load environment
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed, skipping .env file loading")

    # Create benchmark configuration
    config = BenchmarkConfig(
        title=args.title,
        matrix_size=args.matrix_size,
        num_cpus=mp.cpu_count() - 1,  # Reserve one CPU for monitoring
        repeats=args.repeats,
        base_seed=args.base_seed,
        output_dir=Path(args.output_dir),
    )

    # Initialize benchmark
    benchmark = Benchmark(config)

    # Start monitoring
    print("\nInitializing system monitoring...")
    benchmark.setup_monitoring()

    # Print configuration
    print("\nMatrix multiplication benchmark")
    print("Configuration:")
    print(f"  Title: {config.title}")
    print(f"  Repeats: {config.repeats}")
    print(f"  Matrix size: {config.matrix_size}")
    print(f"  CPUs: {config.num_cpus}")
    print(f"  Base seed: {config.base_seed}")
    print(f"  Output directory: {config.output_dir}")

    try:
        # Run benchmark
        print("\nStarting benchmark...")
        results = benchmark.run()

        # Print results
        rounded_results = [round(t, 2) for t in results]
        mean_time = np.mean(results)
        std_time = np.std(results)

        print("\nResults:")
        print(f"Raw timing data: {rounded_results} seconds")
        print(f"Mean time: {mean_time:.2f} seconds")
        print(f"Standard deviation: {std_time:.2f} seconds")

        # Allow time for final data collection
        print("\nCollecting final measurements...")
        time.sleep(10)

        # Generate visualizations and export data
        if benchmark.monitor:
            print("\nGenerating visualizations...")
            benchmark.monitor.plot_data()
            print("\nExporting collected data...")
            benchmark.monitor.export_data()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError during benchmark execution: {e}")
    finally:
        # Clean shutdown
        if benchmark.monitor:
            print("\nShutting down monitoring...")
            benchmark.monitor.stop_event.set()
            time.sleep(2)  # Give monitor thread time to clean up
        print("\nBenchmark complete")


if __name__ == "__main__":
    main()
