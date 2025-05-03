import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from matplotlib.mathtext import MathTextParser
import matplotlib.patches as patches
from PIL import Image, ImageTk
import io
import platform
import csv

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
        title: str = ""
    ):
        self.root = root
        self.root.title("Interactive Time Series Plot")
        
        # Store data
        self.time = time
        self.data1 = data1
        self.data2 = data2
        self.details_text = details_text
        self.current_index = None
        
        # Initialize math text parser
        self.math_parser = MathTextParser('path')
        
        # Create main frame to hold everything
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a vertical PanedWindow with just two sections 
        self.main_paned = tk.PanedWindow(self.main_frame, orient=tk.VERTICAL, 
                                       sashwidth=8, sashrelief="raised",
                                       bg="#888888")
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
            self.plot_container, 
            orient="horizontal", 
            command=self.plot_canvas.xview
        )
        self.plot_v_scrollbar = ttk.Scrollbar(
            self.plot_container, 
            orient="vertical", 
            command=self.plot_canvas.yview
        )
        self.plot_scrollable_frame = ttk.Frame(self.plot_canvas)
        
        # Configure scrolling for plot
        self.plot_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.plot_canvas.configure(
                scrollregion=self.plot_canvas.bbox("all")
            )
        )
        
        # Create window for plot content
        self.plot_canvas.create_window(
            (0, 0), 
            window=self.plot_scrollable_frame, 
            anchor="nw"
        )
        self.plot_canvas.configure(
            xscrollcommand=self.plot_h_scrollbar.set,
            yscrollcommand=self.plot_v_scrollbar.set
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
        
        # Plot data
        self.ax.plot(time, data1, 'o-', color='#1f77b4', linewidth=2, 
                    markersize=6, label=series1_name)
        self.ax.plot(time, data2, 'o-', color='#ff7f0e', linewidth=2, 
                    markersize=6, label=series2_name)
        
        # Configure plot
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_xlim(time.min() - 0.5, time.max() + 0.5)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Pack canvas with a minimum size to ensure scrolling works
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().config(width=600, height=400)
        
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
            self.button_frame, 
            text="Time Steps:", 
            font=('Arial', 11, 'bold')
        )
        self.time_step_label.pack(side=tk.LEFT, padx=(10, 5))
        
        # Create buttons for each time point
        for i, t in enumerate(time):
            btn = ttk.Button(
                self.button_frame,
                text=f"t={t}",
                command=lambda idx=i: self.show_details(idx),
                width=3
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # Add a thick separator after time steps
        sep = ttk.Separator(self.bottom_frame, orient='horizontal', style='TSeparator')
        sep.pack(fill=tk.X, pady=3)
        
        # Create a horizontal PanedWindow for the two detail sections
        self.details_paned = tk.PanedWindow(self.bottom_frame, orient=tk.HORIZONTAL, 
                                          sashwidth=8, sashrelief="raised",
                                          bg="#888888")
        self.details_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === Event Details Section ===
        self.event_container_frame = ttk.Frame(self.details_paned)
        
        self.event_container = ttk.Frame(self.event_container_frame)
        self.event_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add separate title label
        self.event_title = ttk.Label(
            self.event_container,
            text="Event Details",
            font=('Arial', 20, 'bold')
        )
        self.event_title.pack(anchor='w', padx=5, pady=5)
        
        # Create canvas and scrollbar for event details
        self.event_canvas = tk.Canvas(self.event_container)
        self.event_scrollbar = ttk.Scrollbar(
            self.event_container, 
            orient="vertical", 
            command=self.event_canvas.yview
        )
        self.event_scrollable_frame = ttk.Frame(self.event_canvas)
        
        # Configure scrolling for event details
        self.event_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.event_canvas.configure(
                scrollregion=self.event_canvas.bbox("all")
            )
        )
        
        # Create window for event details content
        self.event_canvas.create_window(
            (0, 0), 
            window=self.event_scrollable_frame, 
            anchor="nw"
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
            font=('Arial', 11),
            wraplength=300,
            justify='center'
        )
        self.event_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # === Salient Likelihood Term Section ===
        self.latex_container_frame = ttk.Frame(self.details_paned)
        
        self.latex_container = ttk.Frame(self.latex_container_frame)
        self.latex_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add separate title label
        self.latex_title = ttk.Label(
            self.latex_container,
            text="Salient Likelihood Term", 
            font=('Arial', 20, 'bold')
        )
        self.latex_title.pack(anchor='w', padx=5, pady=5)
        
        # Create canvas and scrollbar for latex details
        self.latex_canvas = tk.Canvas(self.latex_container)
        self.latex_scrollbar = ttk.Scrollbar(
            self.latex_container, 
            orient="vertical", 
            command=self.latex_canvas.yview
        )
        self.latex_scrollable_frame = ttk.Frame(self.latex_canvas)
        
        # Configure scrolling for latex details
        self.latex_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.latex_canvas.configure(
                scrollregion=self.latex_canvas.bbox("all")
            )
        )
        
        # Create window for latex content
        self.latex_canvas.create_window(
            (0, 0), 
            window=self.latex_scrollable_frame, 
            anchor="nw"
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
        self.details_paned.add(self.event_container_frame, minsize=200, stretch="always")
        self.details_paned.add(self.latex_container_frame, minsize=200, stretch="always")
        
        # Add only two main sections to the vertical main paned window
        self.main_paned.add(self.plot_frame, minsize=200, stretch="always")
        self.main_paned.add(self.bottom_frame, minsize=230, stretch="always")
        
        # Configure canvas resize for all scrollable areas
        self.plot_canvas.bind('<Configure>', self._on_plot_canvas_configure)
        self.event_canvas.bind('<Configure>', self._on_event_canvas_configure)
        self.latex_canvas.bind('<Configure>', self._on_latex_canvas_configure)
        
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
                delta = -int(event.delta/120)
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
                delta = -int(event.delta/120)
            elif system == "Darwin":  # macOS
                delta = -event.delta
            elif hasattr(event, 'num'):
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
                canvas.bind_all("<Shift-MouseWheel>", _on_horizontal_mousewheel, add="+")
            elif system == "Linux":
                # Use Shift+Button-4/5 for horizontal scrolling on Linux
                canvas.bind_all("<Shift-Button-4>", _on_horizontal_mousewheel, add="+")
                canvas.bind_all("<Shift-Button-5>", _on_horizontal_mousewheel, add="+")
        
    def _on_plot_canvas_configure(self, event):
        # Update the width of the scrollable frame when the canvas is resized
        if self.plot_canvas.find_all():
            self.plot_canvas.itemconfig(
                self.plot_canvas.find_all()[0], 
                width=event.width
            )
        
    def _on_event_canvas_configure(self, event):
        # Update the width of the scrollable frame when the canvas is resized
        if self.event_canvas.find_all():
            self.event_canvas.itemconfig(
                self.event_canvas.find_all()[0], 
                width=event.width
            )
        
    def _on_latex_canvas_configure(self, event):
        # Update the width of the scrollable frame when the canvas is resized
        if self.latex_canvas.find_all():
            self.latex_canvas.itemconfig(
                self.latex_canvas.find_all()[0], 
                width=event.width
            )
        
    def render_latex(self, latex_str, dpi=120):
        # Create a new figure for rendering LaTeX
        fig = Figure(figsize=(8, 0.8))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        
        # Remove axes
        ax.set_axis_off()
        
        # Render the LaTeX text
        ax.text(0.5, 0.5, latex_str, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12)
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                   pad_inches=0.1, transparent=True)
        plt.close(fig)
        
        # Convert buffer to image
        buf.seek(0)
        img = Image.open(buf)
        return ImageTk.PhotoImage(img)
    
    def show_text(self, text):
        # Render and display the text
        self.current_image = self.render_latex(text)
        self.latex_label.configure(image=self.current_image)
        
    def show_details(self, index):
        self.current_index = index
        # Update both event details and LaTeX formula
        event_detail, latex_formula = self.details_text[index]
        self.event_text.configure(text=event_detail)
        self.show_text(latex_formula)
        
def create_interactive_plot(
    time: np.ndarray,
    data1: np.ndarray,
    data2: np.ndarray,
    details_text: list,
    series1_name: str = "Series 1",
    series2_name: str = "Series 2",
    title: str = "Time Series Plot"
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
        title
    )
    return root

# Example usage
if __name__ == "__main__":
    model = "sob"
    dataset = "ToMi-1st"
    episode = "244"
    with open(f'../results/probs/{model}_{dataset}_{episode}.csv', mode="r") as file:
        reader = csv.reader(file)
        columns = next(reader)
        chunks_probs = [(rows[0], rows[1]) for rows in reader]
    
    choices = eval(columns[1].split('(')[1].split(')')[0])

    with open(f'../results/node_results/{model}_{dataset}_{episode}_back0_reduce1.csv', mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        node_results = [(rows[1], rows[2]) for rows in reader]
    print(node_results)
    # quit()

    time = np.arange(8)
    data1 = np.random.uniform(0.3, 0.9, size=8)
    data2 = np.random.uniform(0.1, 0.7, size=8)
    
    # Example text for each time point (event detail, LaTeX formula)
    details_text = [
        (
            f"At time step {i}, the model detected an important event. This text can be long and will scroll if needed. The scrolling functionality is enabled for both panels independently.",
            f"$\\mathcal{{L}}_{{t={i}}} = \\frac{{1}}{{N}} \\sum_{{i=1}}^{{N}} (y_i - \\hat{{y}}_i)^2$"
        )
        for i in range(8)
    ]
    
    # Create and run the application
    root = create_interactive_plot(
        time=time,
        data1=data1,
        data2=data2,
        details_text=details_text,
        series1_name="Model A",
        series2_name="Model B",
        title="Posterior Probability Over Time"
    )
    root.mainloop() 