# Electrostim Data Analysis

This is a project for processing EMG Data received from the Electrostim Experiment
#

A research-focused pipeline for analyzing electrophysiological data related to **spinal cord excitability**, **H-reflexes**, **M-waves**, and the effects of **TMS** on motor output.

## Overview

This project contains code for processing, visualizing, and analyzing electrophysiological recordings collected during stimulation experiments. It is designed to support workflows involving:

- H-reflex and M-wave detection
- Recruitment curve generation
- Thresholding analysis
- EMG signal processing
- Comparisons of pre/post stimulation excitability
- Visualization of response amplitudes and timing metrics

## Project Goals

The main goals of this repository are to:

- Quantify changes in spinal cord excitability
- Process EMG recordings from stimulation experiments
- Extract useful physiological measures such as:
  - Peak-to-peak amplitude
  - Time to peak
  - H-max
  - M-max
  - Recruitment curve characteristics
- Create figures and summary outputs for interpretation and downstream analysis

## Features

- Read and process raw electrophysiology files
- Detect stimulation events
- Segment EMG responses
- Compute H-reflex and M-wave metrics
- Generate recruitment curves
- Save summary CSV outputs


## Repository Structure

```text
Electrostim/
│
├── README.md                  # Project documentation
├── Hreflex.py                 # Main analysis/processing script
├── initialh.py                 # single grid view
├── HreflexAux.py              # Analysis when using AUX channels for EMG collection
├── quickrecruit.py             # Visualize specific EMG channels
