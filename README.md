# Gemma4 Robot

Gemma4 Robot is a hardware project for helping a person build healthier daily
habits through both physical exercise and mental workout with science and
mathematics problems. It combines a small local device, a camera, pose
estimation, a voice assistant, and a pen plotter. The goal is to coach both
physical activity and intellectual work in a way that feels concrete: the system
can watch movement, talk with the user, create paper tasks, inspect the user's
work, and write feedback back onto paper.

This repository is an entry for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon).
Gemma 4 is the central brain of the system: it drives the assistant, helps turn
goals into tasks, reads student work from camera images, grades reasoning, and
decides what feedback the hardware should give back to the user.

## What It Does

The project has two main parts.

First, it helps with physical exercise. A camera watches the user and a custom
pose-estimation runtime detects body landmarks. The system can use those
landmarks to count squats, push-ups, jumps, and other exercises. This lets the
device act as a small physical-activity coach: it can count repetitions, notice
whether the user is moving, and help make exercise more regular.

Second, it works as an assistant the user can talk with. The assistant can
discuss physical activity, daily routines, well-being, and learning work. It is
not meant to be a medical system; it is a practical coach for movement,
reflection, and study habits.

## Paper Learning Loop

The system is also connected to a plotter: a printer-like mechanism that moves a
pen over paper.

That enables a paper workflow:

1. The assistant creates a task for the student, such as a worksheet.
2. The plotter writes the worksheet on paper.
3. The student solves the task on that sheet.
4. A camera attached to the device takes a photo of the completed work.
5. Gemma 4 reads and grades the work.
6. The plotter writes marks and feedback directly onto the paper.

For example, Gemma 4 could create an Olympiad-style mathematics worksheet, such
as a UK Junior Mathematical Challenge practice question, let the student do the
working by hand, then grade the answer and mark the page.

This connects intellectual work to the physical world: the student writes on
paper, the device reads the work, and the pen plotter writes feedback back onto
the same page.

## Pose Estimation Runtime

The current low-level pose work lives in:

- [`pose_estimation/`](pose_estimation/)
- [`pose_estimation/REPORT.md`](pose_estimation/REPORT.md)

The pose runtime is a custom Raspberry Pi 3B+ NEON implementation of the
MediaPipe Pose Landmarker Lite computation. It is designed to be small and fast
on the Pi without linking TensorFlow Lite, LiteRT, MediaPipe, OpenCV, NumPy, or
XNNPACK into the deployed runtime.

The current best tracked-camera result on the Pi is about `113 ms` per frame,
or about `8.8 FPS`, after the detector has acquired the person. That is the
important steady-state mode for exercise counting.

## Project Direction

The project is meant to be a real local hardware assistant, not just a demo.
The core ideas are:

- local camera-based exercise counting,
- Gemma 4 as the central conversational coach for activity, well-being, and
  learning,
- paper-based learning tasks,
- camera-based grading,
- pen-plotter feedback directly on paper.

Together, these make a system that coaches both physical health and intellectual
work.
