import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
import numpy as np
from matplotlib.mathtext import MathTextParser
import matplotlib.patches as patches
from PIL import Image, ImageTk
import io
import json
import platform
import csv
import pandas as pd
from DataLoader import *
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class InteractiveTimeSeriesPlot:
    def __init__(
        self,
        root,
        time: np.ndarray,
        data1: np.ndarray,
        data2: np.ndarray,
        details_text: list,  # List of tuples (event_detail, latex_formula)
        series1_name: str = "Series 1",
        series2_name: str = "Series 2",
        title: str = "",
        story_question: str = "",
    ):
        self.root = root
        self.root.title("Interactive Time Series Plot")

        # Store data
        self.time = time
        self.data1 = data1
        self.data2 = data2
        self.details_text = details_text
        self.current_index = None
        self.story_question = story_question

        # highlight
        self.highlighting = []

        # Initialize math text parser
        self.math_parser = MathTextParser("path")

        # Create main frame to hold everything
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a vertical PanedWindow with just two sections
        self.main_paned = tk.PanedWindow(
            self.main_frame,
            orient=tk.VERTICAL,
            sashwidth=8,
            sashrelief="raised",
            bg="#888888",
        )
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # === UPPER SECTION (PLOT) ===
        # Create frame for plot
        self.plot_frame = ttk.Frame(self.main_paned)

        # Create LabelFrame for plot content
        self.plot_container = ttk.Frame(self.plot_frame)
        self.plot_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create canvas and scrollbars for plot
        self.plot_canvas = tk.Canvas(self.plot_container)
        self.plot_h_scrollbar = ttk.Scrollbar(
            self.plot_container, orient="horizontal", command=self.plot_canvas.xview
        )
        self.plot_v_scrollbar = ttk.Scrollbar(
            self.plot_container, orient="vertical", command=self.plot_canvas.yview
        )
        self.plot_scrollable_frame = ttk.Frame(self.plot_canvas)

        # Configure scrolling for plot
        self.plot_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.plot_canvas.configure(
                scrollregion=self.plot_canvas.bbox("all")
            ),
        )

        # Create window for plot content
        self.plot_canvas.create_window(
            (0, 0), window=self.plot_scrollable_frame, anchor="nw"
        )
        self.plot_canvas.configure(
            xscrollcommand=self.plot_h_scrollbar.set,
            yscrollcommand=self.plot_v_scrollbar.set,
        )

        # Pack plot components - scrollbars and canvas
        self.plot_h_scrollbar.pack(side="bottom", fill="x")
        self.plot_v_scrollbar.pack(side="right", fill="y")
        self.plot_canvas.pack(side="left", fill="both", expand=True)

        # Create plot inside scrollable frame
        self.chart_frame = ttk.Frame(self.plot_scrollable_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)

        self.canvas.draw()

        # Plot data
        self.ax.plot(
            time,
            data1,
            "o-",
            color="#1f77b4",
            linewidth=2,
            markersize=6,
            label=series1_name,
        )
        self.ax.plot(
            time,
            data2,
            "o-",
            color="#ff7f0e",
            linewidth=2,
            markersize=6,
            label=series2_name,
        )

        # Configure plot
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_xlim(time.min() - 0.5, time.max() + 0.5)
        self.ax.set_xlabel("Timestamp")
        self.ax.set_ylabel("Probability")
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        # integer ticks
        self.ax.xaxis.set_major_locator(MultipleLocator(1))

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        # Pack canvas with a minimum size to ensure scrolling works
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().config(width=800, height=400)

        # === LOWER SECTION (TIME STEPS + DETAILS) ===
        self.bottom_frame = ttk.Frame(self.main_paned)

        # Style for separator
        style = ttk.Style()
        style.configure("TSeparator", background="#888888", relief="groove")

        # Create time steps section first
        self.timesteps_frame = ttk.Frame(self.bottom_frame)
        self.timesteps_frame.pack(fill=tk.X, pady=5)

        # Create buttons frame for time steps
        self.button_frame = ttk.Frame(self.timesteps_frame)
        self.button_frame.pack(fill=tk.X, pady=5)

        # Label for time steps
        self.time_step_label = ttk.Label(
            self.button_frame, text="Time Steps:", font=("Arial", 14, "bold")
        )
        self.time_step_label.pack(side=tk.LEFT, padx=(10, 5))

        # Create buttons for each time point
        for i, t in enumerate(time):
            btn = ttk.Button(
                self.button_frame,
                text=f"t={t}",
                command=lambda idx=i: self.show_details(idx),
                width=3,
            )
            btn.pack(side=tk.LEFT, padx=2)

        self.time_step_label1 = ttk.Label(
            self.timesteps_frame,
            text=self.story_question,
            font=(
                "Arial",
                20,
            ),
            # wraplength=1400,
        )
        self.time_step_label1.pack(anchor="w", padx=(10, 5), pady=(5, 0))
        self._make_wrap_follow_width(self.time_step_label)
        self._make_wrap_follow_width(self.time_step_label1)

        # Add a thick separator after time steps
        sep = ttk.Separator(self.bottom_frame, orient="horizontal", style="TSeparator")
        sep.pack(fill=tk.X, pady=3)

        # Create a horizontal PanedWindow for the two detail sections
        self.details_paned = tk.PanedWindow(
            self.bottom_frame,
            orient=tk.HORIZONTAL,
            sashwidth=8,
            sashrelief="raised",
            bg="#888888",
        )
        self.details_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === Event Details Section ===
        self.event_container_frame = ttk.Frame(self.details_paned)

        self.event_container = ttk.Frame(self.event_container_frame)
        self.event_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add separate title label
        self.event_title = ttk.Label(
            self.event_container, text="Timestamp Details", font=("Arial", 20, "bold")
        )
        self.event_title.pack(anchor="w", padx=5, pady=5)

        # Create canvas and scrollbar for event details
        self.event_canvas = tk.Canvas(self.event_container)
        self.event_scrollbar = ttk.Scrollbar(
            self.event_container, orient="vertical", command=self.event_canvas.yview
        )
        self.event_scrollable_frame = ttk.Frame(self.event_canvas)

        # Configure scrolling for event details
        self.event_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.event_canvas.configure(
                scrollregion=self.event_canvas.bbox("all")
            ),
        )

        # Create window for event details content
        self.event_canvas.create_window(
            (0, 0), window=self.event_scrollable_frame, anchor="nw"
        )
        self.event_canvas.configure(yscrollcommand=self.event_scrollbar.set)

        # Pack event details components
        self.event_scrollbar.pack(side="right", fill="y")
        self.event_canvas.pack(side="left", fill="both", expand=True)

        # Content for event details
        self.event_content = ttk.Frame(self.event_scrollable_frame)
        self.event_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.event_text = ttk.Label(
            self.event_content,
            text="Select a time point to show details",
            font=("Arial", 20),
            wraplength=400,
            justify="center",
        )
        self.event_text.pack(fill=tk.BOTH, expand=True, pady=10)
        self._make_wrap_follow_width(self.event_text)

        # === Salient Likelihood Term Section ===
        self.latex_container_frame = ttk.Frame(self.details_paned)

        self.latex_container = ttk.Frame(self.latex_container_frame)
        self.latex_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add separate title label
        self.latex_title = ttk.Label(
            self.latex_container,
            text="Salient Likelihood Term",
            font=("Arial", 20, "bold"),
        )
        self.latex_title.pack(anchor="w", padx=5, pady=5)
        self._make_wrap_follow_width(self.latex_title)

        # Create canvas and scrollbar for latex details
        self.latex_canvas = tk.Canvas(self.latex_container)
        self.latex_scrollbar = ttk.Scrollbar(
            self.latex_container, orient="vertical", command=self.latex_canvas.yview
        )
        self.latex_scrollable_frame = ttk.Frame(self.latex_canvas)

        # Configure scrolling for latex details
        self.latex_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.latex_canvas.configure(
                scrollregion=self.latex_canvas.bbox("all")
            ),
        )

        # Create window for latex content
        self.latex_canvas.create_window(
            (0, 0), window=self.latex_scrollable_frame, anchor="nw"
        )
        self.latex_canvas.configure(yscrollcommand=self.latex_scrollbar.set)

        # Pack latex components
        self.latex_scrollbar.pack(side="right", fill="y")
        self.latex_canvas.pack(side="left", fill="both", expand=True)

        # Content for latex details
        self.latex_content = ttk.Frame(self.latex_scrollable_frame)
        self.latex_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a label for rendered LaTeX
        self.latex_label = ttk.Label(self.latex_content)
        self.latex_label.pack(fill=tk.BOTH, expand=True, pady=10)

        # Render initial LaTeX
        self.show_text("$\\text{Select a time point to show details}$")

        # ADD FRAMES TO PANEDWINDOWS
        # Add components to the horizontal details paned window

        self.details_paned.add(
            self.event_container_frame, minsize=200, stretch="always"
        )
        self.details_paned.add(
            self.latex_container_frame, minsize=600, stretch="always"
        )

        # Add only two main sections to the vertical main paned window
        self.main_paned.add(self.plot_frame, minsize=200, stretch="always")
        self.main_paned.add(self.bottom_frame, minsize=230, stretch="always")

        # Configure canvas resize for all scrollable areas
        self.plot_canvas.bind("<Configure>", self._on_plot_canvas_configure)
        self.event_canvas.bind("<Configure>", self._on_event_canvas_configure)
        self.latex_canvas.bind("<Configure>", self._on_latex_canvas_configure)

        # Bind mousewheel to scrollbars
        self.bind_mousewheel(self.plot_canvas, has_horizontal=True)
        self.bind_mousewheel(self.event_canvas)
        self.bind_mousewheel(self.latex_canvas)

    def bind_mousewheel(self, canvas, has_horizontal=False):
        # Platform detection for appropriate bindings
        system = platform.system()

        # Vertical scrolling handler
        def _on_vertical_mousewheel(event):
            # Normalize delta across platforms
            delta = 0
            if system == "Windows":
                delta = -int(event.delta / 120)
            elif system == "Darwin":  # macOS
                delta = -event.delta
            else:  # Linux
                if event.num == 4:
                    delta = -1
                elif event.num == 5:
                    delta = 1

            if delta != 0:
                canvas.yview_scroll(delta, "units")
                return "break"  # Prevent event propagation

        # Horizontal scrolling handler with shift key
        def _on_horizontal_mousewheel(event):
            delta = 0
            if system == "Windows":
                delta = -int(event.delta / 120)
            elif system == "Darwin":  # macOS
                delta = -event.delta
            elif hasattr(event, "num"):
                if event.num == 4:
                    delta = -1
                elif event.num == 5:
                    delta = 1

            if delta != 0:
                canvas.xview_scroll(delta, "units")
                return "break"  # Prevent event propagation

        # Bind vertical scrolling for all platforms
        if system == "Linux":
            canvas.bind_all("<Button-4>", _on_vertical_mousewheel, add="+")
            canvas.bind_all("<Button-5>", _on_vertical_mousewheel, add="+")
        elif system == "Windows":
            canvas.bind_all("<MouseWheel>", _on_vertical_mousewheel, add="+")
        elif system == "Darwin":  # macOS
            canvas.bind_all("<MouseWheel>", _on_vertical_mousewheel, add="+")

        # Bind horizontal scrolling if needed
        if has_horizontal:
            # Shift+MouseWheel for horizontal scrolling
            if system in ["Windows", "Darwin"]:  # Windows or macOS
                canvas.bind_all(
                    "<Shift-MouseWheel>", _on_horizontal_mousewheel, add="+"
                )
            elif system == "Linux":
                # Use Shift+Button-4/5 for horizontal scrolling on Linux
                canvas.bind_all("<Shift-Button-4>", _on_horizontal_mousewheel, add="+")
                canvas.bind_all("<Shift-Button-5>", _on_horizontal_mousewheel, add="+")

    def _on_plot_canvas_configure(self, event):
        # Update the width of the scrollable frame when the canvas is resized
        if self.plot_canvas.find_all():
            self.plot_canvas.itemconfig(
                self.plot_canvas.find_all()[0], width=event.width
            )

    def _on_event_canvas_configure(self, event):
        # Update the width of the scrollable frame when the canvas is resized
        if self.event_canvas.find_all():
            self.event_canvas.itemconfig(
                self.event_canvas.find_all()[0], width=event.width
            )

    def _on_latex_canvas_configure(self, event):
        # Update the width of the scrollable frame when the canvas is resized
        if self.latex_canvas.find_all():
            self.latex_canvas.itemconfig(
                self.latex_canvas.find_all()[0], width=event.width
            )

    def render_latex(self, latex_str, dpi=120):
        # Create a new figure for rendering LaTeX
        fig = Figure(figsize=(8, 0.8))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)

        # Remove axes
        ax.set_axis_off()

        # Render the LaTeX text
        ax.text(
            0.5,
            0.5,
            latex_str,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=True,
        )
        plt.close(fig)

        # Convert buffer to image
        buf.seek(0)
        img = Image.open(buf)
        return ImageTk.PhotoImage(img)

    def show_text(self, text):
        # Render and display the text
        self.current_image = self.render_latex(text)
        self.latex_label.configure(image=self.current_image)

    def _make_wrap_follow_width(self, lbl: ttk.Label):
        def _resize(event, l=lbl):
            l.configure(wraplength=event.width)

        lbl.bind("<Configure>", _resize)

    def _show_mixed(self, items):
        for w in self.latex_content.winfo_children():
            w.destroy()

        if not isinstance(items, (list, tuple)):
            items = [items]

        for obj in items:
            if isinstance(obj, Figure):
                buf = io.BytesIO()
                obj.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf)
                photo = ImageTk.PhotoImage(img)

                lbl = ttk.Label(self.latex_content, image=photo)
                lbl.image = photo
                lbl.pack(fill=tk.BOTH, expand=True, pady=4)

            elif isinstance(obj, str) and os.path.isfile(obj):
                img = Image.open(obj)
                img.thumbnail((400, 400), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(img)

                lbl = ttk.Label(self.latex_content, image=photo)
                lbl.image = photo
                lbl.pack(fill=tk.BOTH, expand=True, pady=4)

            elif isinstance(obj, str):
                if obj.strip().startswith("$") and obj.strip().endswith("$"):
                    photo = self.render_latex(obj)
                    lbl = ttk.Label(self.latex_content, image=photo)
                    lbl.image = photo
                else:
                    lbl = ttk.Label(
                        self.latex_content,
                        text=obj,
                        font=("Arial", 20),
                        # wraplength=1000,
                        justify="left",
                    )
                lbl.pack(fill=tk.BOTH, expand=True, pady=4)

            elif isinstance(obj, ImageTk.PhotoImage):
                lbl = ttk.Label(self.latex_content, image=obj)
                lbl.image = obj
                lbl.pack(fill=tk.BOTH, expand=True, pady=4)

            else:
                lbl = ttk.Label(self.latex_content, text=str(obj))
                lbl.pack(fill=tk.BOTH, expand=True, pady=4)

    def show_details(self, index):
        left, right_payload = self.details_text[index]
        self.event_text.configure(text=left)
        self._show_mixed(right_payload)

        for h in self.highlighting:
            try:
                h.remove()
            except Exception:
                pass
        self.highlighting.clear()

        h1 = self.ax.scatter(
            [self.time[index]],
            [self.data1[index]],
            s=200,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=5,
        )
        h2 = self.ax.scatter(
            [self.time[index]],
            [self.data2[index]],
            s=200,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=5,
        )
        self.highlighting.extend([h1, h2])
        self.canvas.draw_idle()


def create_interactive_plot(
    time: np.ndarray,
    data1: np.ndarray,
    data2: np.ndarray,
    details_text: list,
    series1_name: str = "Series 1",
    series2_name: str = "Series 2",
    title: str = "Time Series Plot",
    story_question: str = "",
):
    root = tk.Tk()
    # Set minimum size for the window
    root.minsize(800, 600)
    app = InteractiveTimeSeriesPlot(
        root,
        time,
        data1,
        data2,
        details_text,
        series1_name,
        series2_name,
        title,
        story_question,
    )
    return root


# Example usage
if __name__ == "__main__":
    model = "sobag"
    # model = "automated"
    dataset = "MMToM-QA"
    episode = "300"
    # dataset = "BigToM_bbfb"
    # episode = "6"
    direction = "1"
    larger_path = "../"

    data = load_full_dataset(dataset)
    d = data[int(episode)]
    if "MuMa" in dataset:
        story, question, choices, correct_answer, video_id = d
    elif "MMToM" in dataset:
        story, question, choices, correct_answer, states, actions = d
    else:
        story, question, choices, correct_answer = d

    if correct_answer == "A":
        correct_answer = choices[0]
    elif correct_answer == "B":
        correct_answer = choices[1]
    elif correct_answer == "C":
        correct_answer = choices[2]

    with open(
        f"{larger_path}results/probs/{model}_{dataset}_{episode}.csv",
        mode="r",
    ) as file:
        reader = csv.reader(file)
        columns = next(reader)
        chunks_probs = [(rows[0], rows[1]) for rows in reader]

    choices = eval(columns[1].split("(")[1].split(")")[0])


    with open(
        f"{larger_path}results/node_results/{model}_{dataset}_{episode}_back1_reduce1.csv",
        mode="r",
    ) as file:
        reader = csv.reader(file)
        next(reader)
        node_results = [(rows[1], rows[2]) for rows in reader]
    print(node_results)
    # quit()

    if "MMToM" not in dataset:
        df_middle = pd.read_csv(
            f"{larger_path}results/middle/{model}_{dataset}_{episode}.csv"
        )
        total_num_steps = len(df_middle)
    else:
        total_num_steps = len(actions)

    time = np.arange(total_num_steps)
    # plot the overall probabilities
    hyp_a = [np.nan] * total_num_steps
    hyp_b = [np.nan] * total_num_steps
    # backward run
    if direction == "1":
        counter = total_num_steps - 1
        df_probs = pd.read_csv(
            # f"{larger_path}results/probs/{model}_{dataset}_{episode}.csv"
            f"{larger_path}results/probs/{model}_{dataset}_{episode}.csv"
        )
        for index, row in df_probs.iloc[::-1].iterrows():
            print(index, row)
            val = row[df_probs.columns[1]]
            if ',' not in val:
                val = val.replace(' ', ',')
            prob_a, prob_b = (
                eval(val)[0],
                eval(val)[1],
            )
            hyp_a[counter] = prob_a
            hyp_b[counter] = prob_b
            counter -= 1

    # forward run
    elif direction == "0":
        counter = 0
        df_probs = pd.read_csv(
            # f"{larger_path}results/probs/{model}_{dataset}_{episode}.csv"
            f"{larger_path}results/probs/{model}_{dataset}_{episode}.csv"
        )
        for index, row in df_probs.iterrows():
            prob_a, prob_b = (
                eval(row[df_probs.columns[1]])[0],
                eval(row[df_probs.columns[1]])[1],
            )
            hyp_a[counter] = prob_a
            hyp_b[counter] = prob_b
            counter += 1


    with open(f"{larger_path}results/parsed_result/{model}_{dataset}_{episode}.json") as f:
        parsed_result = json.load(f)
    with open(f"{larger_path}results/NLD_descriptions/{model}_{dataset}_{episode}.json") as f:
        NLDs = json.load(f)
    # Example text for each time point (event detail, LaTeX formula)
    details_text = []
    for i in range(total_num_steps):
        joints = []
        joint_str = ""
        joint_val = 0
        if hyp_a[i] > hyp_b[i]:
            curr_max_hyp = choices[0]
            curr_max_prob = hyp_a[i]
        else:
            curr_max_hyp = choices[1]
            curr_max_prob = hyp_b[i]
        df_node = pd.read_csv(
            f"{larger_path}results/node_results/{model}_{dataset}_{episode}_back1_reduce1.csv"
        )
        df_node = df_node[df_node["Time"] == i]
        node_info = ""
        joint_info = ""
        local_conds = []
        normalization_constant = 0
        action_info = ""
        action_liks = []
        df_node_action = df_node[df_node["Node"].str.contains(f"Action_{i}", na=False)]
        df_node_action = df_node_action.sort_values("Node", ascending=True)
        action_liks = list(df_node_action["Likelihood"])

        for c in range(1, len(choices) + 1):
            if True: 
                local_cond = 1
                node_info += f"{parsed_result['inf_var_name']} Hypothesis {c}: {choices[c-1]} \n"
                if str(i) in NLDs:
                    # print(NLDs[str(i)][0])
                    node_info += f"One of the most salient joint probability is {NLDs[str(i)][0][0][1]}, calculated using these values and hypotheses:\n"
                    node_info += NLDs[str(i)][0][0][0]
                # df_sub_node = df_node[
                #     df_node["Node"].str.contains(f"Belief_{i}_{c}", na=False)
                #     | df_node["Parent node"].str.contains(f"Belief_{i}_{c}", na=False)
                # ]
                # for index, row in df_sub_node.iterrows():
                #     likelihood = f'P({row["Node"]}={row["Node value"]} | {row["Parent node"]}={row["Parent node value"]}) = {round(row["Likelihood"], 3)} \n'
                #     print(likelihood)
                #     if likelihood not in node_info:
                #         node_info += likelihood
                #     local_cond *= row["Likelihood"]

            if (
                "BigToM_bbfb" in dataset or "BigToM_bbtb" in dataset
            ):  # TODO: fix the joint probability calculations
                if i == 0:
                    prior_a = 0
                    prior_b = 0
                    # answer choice 1
                    df_sub_node = df_node[
                        df_node["Node"].str.contains(f"Belief_{i}_{c}", na=False)
                    ]
                    belief_cond_obs_prob = round(list(df_sub_node["Likelihood"])[0], 3)
                    df_sub_node = df_node[
                        df_node["Parent node"].str.contains(f"Belief_{i}_{c}", na=False)
                        & df_node["Node"].str.contains(f"Action_{i}_1", na=False)
                    ]
                    action_cond_obs_prob = round(list(df_sub_node["Likelihood"])[0], 3)
                    joint_str += f"{action_cond_obs_prob} * {belief_cond_obs_prob}"
                    joint_val += action_cond_obs_prob * belief_cond_obs_prob

                    joint_info += f"joint prob for P(b_{c}, o, g, a): P(b_{c}|o) * P(a|g,b_{c}) * P(g) * P(o)= {joint_str} = {round(joint_val, 3)} \n"
                    joints.append(joint_val)
                    joint_str = ""
                    joint_val = 0

                if i > 0:
                    prior_a = hyp_a[i - 1]
                    prior_b = hyp_b[i - 1]
                    # answer choice 1
                    if c == 1:
                        df_sub_node_a = df_node[
                            df_node["Node"].str.contains(f"Belief_{i}_{c}", na=False)
                            & df_node["Parent node"].str.contains(
                                f"Belief_{i-1}_{c}", na=False
                            )
                        ]
                        df_sub_node_b = df_node[
                            df_node["Node"].str.contains(f"Belief_{i}_{c}", na=False)
                            & df_node["Parent node"].str.contains(
                                f"Belief_{i-1}_2", na=False
                            )
                        ]
                        belief_cond_prev_belief_a = list(df_sub_node_a["Likelihood"])[0]
                        belief_cond_prev_belief_b = list(df_sub_node_b["Likelihood"])[0]
                    # answer choice 2
                    else:
                        df_sub_node_a = df_node[
                            df_node["Node"].str.contains(f"Belief_{i}_{c}", na=False)
                            & df_node["Parent node"].str.contains(
                                f"Belief_{i-1}_1", na=False
                            )
                        ]

                        df_sub_node_b = df_node[
                            df_node["Node"].str.contains(f"Belief_{i}_{c}", na=False)
                            & df_node["Parent node"].str.contains(
                                f"Belief_{i-1}_{c}", na=False
                            )
                        ]
                        belief_cond_prev_belief_a = list(df_sub_node_a["Likelihood"])[0]
                        belief_cond_prev_belief_b = list(df_sub_node_b["Likelihood"])[0]
                    df_sub_node = df_node[
                        df_node["Parent node"].str.contains(f"Belief_{i}_{c}", na=False)
                        & df_node["Node"].str.contains(f"Action_{i}_1", na=False)
                    ]
                    action_cond_obs_prob = round(list(df_sub_node["Likelihood"])[0], 3)
                    joint_str += f"{round(action_cond_obs_prob, 3)} * ({round(prior_a, 3)} * {round(belief_cond_prev_belief_a, 3)} + {round(prior_b, 3)} * {round(belief_cond_prev_belief_b, 3)})"
                    joint_val += action_cond_obs_prob * (
                        prior_a * belief_cond_prev_belief_a
                        + prior_b * belief_cond_prev_belief_b
                    )
                    if c == 0:
                        alter_c = 1
                    else:
                        alter_c = 0
                    joint_info += f"joint prob for P(b_{c}, o, g, a): P(o) * P(g) * P(a|b,g) * Σ(P(prev b_i) * P(b_{c}|o, prev b_i)) = {joint_str} = {round(joint_val, 3)} \n"
                    joints.append(joint_val)
                    joint_str = ""
                    joint_val = 0
                node_info += "---------- \n"

        # max_action = max(action_liks)
        # if max_action == action_liks[0]:
        #     action_info += f"• Action likelihood is higher for {choices[0]} ({round(action_liks[0], 3)} > {round(action_liks[1], 3)}) \n"
        # else:
        #     action_info += f"• Action likelihood is higher for {choices[1]} ({round(action_liks[1], 3)} > {round(action_liks[0], 3)}) \n"
        # node_info = action_info + "---------- \n" + node_info
        # action_liks = []

        # if len(joints) > 0:
        #     max_marg = max(joints)
        #     if max_marg == joints[0]:
        #         joint_info = (
        #             f"• joint priors are higher for {choices[0]} \n" + joint_info
        #         )
        #     else:
        #         joint_info = (
        #             f"• joint priors are higher for {choices[1]} \n" + joint_info
        #         )
        node_info = joint_info + "---------- \n" + node_info

        local_conds.append(local_cond)
        normalization_constant += local_cond

        with open(
            f"{larger_path}results/metrics/{model}_{dataset}_{episode}_back1_reduce1_metrics.json",
            "r",
        ) as f:
            metrics_data = json.load(f)

        if "Model Record" in metrics_data:
            assigned_models = metrics_data["Model Record"][f"Question {episode}"][
                "Assigned models"
            ]
        else:
            assigned_models = []
            for x in model:
                if x == "s":
                    assigned_models.append("State")
                if x == "a":
                    assigned_models.append("Action")
                if x == "b":
                    assigned_models.append("Belief")
                if x == "o":
                    assigned_models.append("Observation")
                if x == "g":
                    assigned_models.append("Goal")

        if "MMToM" not in dataset:
            left = f'At time step {i}, {list(df_middle["Chunk"])[i]}\n\nThe proposed model is {assigned_models[str(i)]}\n\n•P({choices[0]}) = {round(hyp_a[i], 3)} \n•P({choices[1]}) = {round(hyp_b[i], 3)} '
        else:
            left = f"At time step {i}, {states[i]} \nThe proposed model is {assigned_models}\n\n•P({choices[0]}) = {round(hyp_a[i], 3)} \n•P({choices[1]}) = {round(hyp_b[i], 3)} \nThe extracted variables are: {NLDs[str(i)][1] if str(i) in NLDs else 'Not Inferred'} "
        right_text = (
            f"{node_info}"
            # f"$\\mathcal{{L}}_{{t={i}}}=\\frac{{1}}{{N}}\\sum(y_i-\\hat{{y}}_i)^2$"
        )

        png_base_path = f"{larger_path}results/node_graphs/{model}_{dataset}_{episode}/timestamp_{i}.png"

        right_payload = [
            png_base_path,
            right_text,
        ]

        details_text.append((left, right_payload))
        # details_text.append(
        #     (  # left pane
        #         f"At time step {i}, the hypothesis with the higher probability is {curr_max_hyp} with a probability of {curr_max_prob}. \n Story: {story} \n Question: {question} \n The actual corrrect answer should be {correct_answer}.",
        #         # right pane
        #         f"$\\mathcal{{L}}_{{t={i}}} = \\frac{{1}}{{N}} \\sum_{{i=1}}^{{N}} (y_i - \\hat{{y}}_i)^2$",
        #     )
        # )

    choices = eval(df_probs.columns[1].replace("Probs(", "").replace(")", ""))
    # Create and run the application
    root = create_interactive_plot(
        time=time,
        data1=hyp_a,
        data2=hyp_b,
        details_text=details_text,
        series1_name=choices[0],
        series2_name=choices[1],
        title="",  # Posterior Probability Over Time",
        story_question=f"Story: {story}\nQuestion: {question}\nCorrect Answer: {correct_answer}. \n".replace(
            "..", "."
        ),
    )
    root.mainloop()
