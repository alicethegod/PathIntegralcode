# -*- coding: utf-8 -*-
"""Optimized Publication-Grade Plotting Script

D-Scaling Experiment Data Visualization Script (Optimized for Publication)

This script provides a Graphical User Interface (GUI) to select a CSV file generated
by the D-Scaling experiment. It then processes the data using established reducible
metrics and generates a 2x2 plot that conforms to the style standards of top-tier
journals.

Core Optimizations:
1.  Professional visual style (fonts, colors, line widths).
2.  Use of 'constrained_layout' to automatically optimize plot layout and prevent label overlap.
3.  Intelligent adjustment of text annotation box position and style to avoid obscuring data points.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Core Plotting and Analysis Function (Optimized) ---

def plot_d_scaling_thermodynamics_professional(csv_path):
    """
    Loads D-Scaling data from CSV, calculates reducible metrics, fits curves,
    and generates a professional 2x2 publication-grade plot.
    """
    if not os.path.exists(csv_path):
        messagebox.showerror("Error", f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path).sort_values('data_size_d')

    # --- 1. Configuration ---
    NUM_POINTS_FOR_BASELINE = 5  # Number of data points used to estimate the baseline (inf or 0)

    # --- 2. Calculate Reducible Metrics ---
    # Estimate convergence value (L_inf, S_inf) using the last few points
    L_inf = df['final_test_loss'].tail(NUM_POINTS_FOR_BASELINE).mean()
    df['reducible_loss'] = df['final_test_loss'] - L_inf

    # Assuming 'final_S' and 'final_U' are the counterparts to H'sie and H'tse
    # for the thermodynamic perspective (Entropy S and Internal Energy U)
    S_inf = df['final_S'].tail(NUM_POINTS_FOR_BASELINE).mean()
    df['reducible_S'] = df['final_S'] - S_inf

    # Estimate initial value (U_0) using the first few points
    U_0 = df['final_U'].head(NUM_POINTS_FOR_BASELINE).mean()
    df['reducible_U'] = df['final_U'] - U_0

    # Calculate the ratio U/F (K-Complexity proxy, where F is Free Energy, assuming F is equivalent to the old H'tse/H'sie norm)
    # NOTE: The original script uses final_F. This assumes final_F is calculated elsewhere or is a column in the CSV.
    # If final_F is missing, this line will cause an error. We proceed assuming it exists.
    df['U_F_ratio'] = df['final_U'] / df['final_F']

    # --- 3. Set Visualization Style ---
    # Using professional and aesthetic parameter configurations
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'], # Use classic serif font
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 24,
        'axes.titlesize': 20,
        'mathtext.fontset': 'cm', # Use LaTeX-like math font
        'axes.linewidth': 1.2,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7
    })

    # Use constrained_layout=True to automatically optimize the layout and prevent overlap
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle('Thermodynamic Analysis of D-Scaling', y=1.03)

    x_d = df['data_size_d'].values
    # Define a set of clear, high-contrast colors
    colors = {
        'loss': '#1f77b4', 'U': '#ff7f0e', 'S': '#2ca02c',
        'ratio': '#d62728', 'fit': '#9467bd'
    }

    # --- 4. Fitting Functions ---
    def power_law_fit(x, y):
        mask = (y > 0) & (x > 0) & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 2: return 0, 1, 0, np.full_like(x, np.nan, dtype=float)
        log_x, log_y = np.log10(x[mask]), np.log10(y[mask])
        s, i, r, p, _ = linregress(log_x, log_y)
        r2 = r**2
        fit_curve = 10**(s * np.log10(x) + i)
        return r2, p, s, fit_curve

    def exp_decay_fit(x, y):
        mask = (y > 0) & (x > 0) & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 2: return 0, 1, 0, np.full_like(x, np.nan, dtype=float)
        log_y = np.log(y[mask])
        slope, intercept, r, p, _ = linregress(x[mask], log_y)
        r2 = r**2
        decay_rate = -slope
        fit_curve = np.exp(slope * x + intercept)
        return r2, p, decay_rate, fit_curve

    # --- 5. Generate Plots ---

    # Top Left: Performance Scaling
    ax = axes[0, 0]
    r2, p, s, fit = power_law_fit(x_d, df['reducible_loss'])
    ax.plot(x_d, df['final_test_loss'], 'o', color=colors['loss'], markersize=8, label='Data')
    ax.plot(x_d, fit + L_inf, '--', color=colors['fit'], lw=3, label='Fit')
    ax.set_title('Performance Scaling')
    ax.set_ylabel('Final Test Loss')
    # Optimize text box style and position
    text_box_style = dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7, ec='gray')
    ax.text(0.95, 0.95, f'$R^2={r2:.2f}, p={p:.1e}$\n$L-L_\\infty \\propto D^{{{s:.2f}}}$',
            ha='right', va='top', transform=ax.transAxes, bbox=text_box_style, fontsize=12)

    # Top Right: Internal Energy Scaling
    ax = axes[0, 1]
    r2, p, s, fit = power_law_fit(x_d, df['reducible_U'])
    ax.plot(x_d, df['final_U'], 's', color=colors['U'], markersize=7, label='Data')
    ax.plot(x_d, fit + U_0, '--', color=colors['fit'], lw=3, label='Fit')
    ax.set_title('Internal Energy Scaling')
    ax.set_ylabel('Cognitive Internal Energy ($U$)')
    # Move text box to the top left corner to avoid overlap with the end of the curve
    ax.text(0.05, 0.95, f'$R^2={r2:.2f}, p={p:.1e}$\n$U-U_0 \\propto D^{{{s:.2f}}}$',
            ha='left', va='top', transform=ax.transAxes, bbox=text_box_style, fontsize=12)

    # Bottom Left: Entropy Scaling
    ax = axes[1, 0]
    r2, p, decay_rate, fit = exp_decay_fit(x_d, df['reducible_S'])
    ax.plot(x_d, df['final_S'], '^', color=colors['S'], markersize=8, linestyle='None', label='Data')
    ax.plot(x_d, fit + S_inf, '--', color=colors['fit'], lw=3, label='Fit')
    ax.set_title('Entropy Scaling')
    ax.set_ylabel('Cognitive Entropy ($S$)')
    # Place text box in the blank area after the curve drops
    ax.text(0.95, 0.05, f'$R^2={r2:.2f}, p={p:.1e}$\n$S-S_\\infty \\propto e^{{-{decay_rate:.1e}D}}$',
            ha='right', va='bottom', transform=ax.transAxes, bbox=text_box_style, fontsize=12)

    # Bottom Right: K-Complexity Proxy Scaling
    ax = axes[1, 1]
    ax.plot(x_d, df['U_F_ratio'], 'd-', color=colors['ratio'], markersize=7, lw=2.5)
    ax.set_title('K-Complexity Proxy Scaling')
    ax.set_ylabel(r'U/F Ratio ($U / F$)')
    ax.axhline(0, color='grey', linestyle=':', linewidth=1.5) # Use dashed line to be less intrusive
    ax.set_yscale('linear') # Keep linear scale

    # --- 6. Final Formatting and Saving ---
    for ax in axes.flatten():
        ax.set_xlabel('Dataset Size ($D$)')
        ax.set_xscale('log')
        if ax != axes[1, 1]: # All plots except the bottom right one use a log y-axis
             ax.set_yscale('log')

        # Remove top and right borders for a more modern style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', length=6, width=1.2)
        ax.tick_params(axis='both', which='minor', length=4, width=0.8)

    # Save the file
    output_filename = os.path.splitext(csv_path)[0] + '_professional_plot.png'
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
        plt.show()
    except Exception as e:
        messagebox.showerror("Save Failed", f"Could not save the plot:\n{e}")


# --- GUI Functionality ---

def select_file_and_plot(root):
    """Opens a file dialog to select the CSV, then calls the plotting function."""
    filepath = filedialog.askopenfilename(
        title="Select a D-Scaling CSV file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if filepath:
        root.withdraw() # Hide the GUI window while plotting
        try:
            # Call the optimized new function
            plot_d_scaling_thermodynamics_professional(filepath)
            messagebox.showinfo("Success", "Plot generated and saved.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during plotting:\n{e}")
        finally:
            root.destroy() # Close the application when finished

def main_gui():
    """Creates and runs the main Tkinter GUI."""
    root = tk.Tk()
    root.title("D-Scaling Professional Plotting Tool")
    root.geometry("400x150")

    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(expand=True, fill=tk.BOTH)

    label = tk.Label(frame, text="Please select the CSV data file generated by the D-Scaling experiment.", wraplength=350)
    label.pack(pady=10)

    plot_button = tk.Button(frame, text="Select CSV and Generate Plot", command=lambda: select_file_and_plot(root))
    plot_button.pack(pady=10)

    root.mainloop()

# --- Main Execution Block ---

if __name__ == '__main__':
    main_gui()
