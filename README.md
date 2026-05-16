# Gemma4 Robot

Gemma4 Robot is a hardware project for the future of learning and healthy
habits. Its primary feature is a paper-based Science and Mathematics coach: the
system creates worksheets, writes them on paper with a pen plotter, reads the
student's handwritten work with a camera, uses Gemma 4 to grade the reasoning,
and writes feedback directly back onto the same paper. Its second feature is an
active sports coach that uses camera-based pose estimation to count exercises
such as squats, push-ups, and jumps.

The device is meant to be controlled naturally, without a keyboard. The user
talks to it through a microphone, shows work and movement through cameras, and
gets responses through the screen, audio, and marks written by the plotter. A
single physical button can be used for simple control.

This repository is an entry for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon).
Gemma 4 is the central brain of the system: it drives the assistant, turns
learning goals into tasks, reads student work from camera images, grades
reasoning, decides what feedback should be written back to paper, and supports
the physical-activity coach.

## What It Does

The project has two main parts.

First, it is a science and mathematics learning coach. The system can create a
worksheet, write it on paper, let the student solve it by hand, photograph the
completed work, use Gemma 4 to grade the reasoning, and then write marks and
feedback directly onto the sheet. The point is to keep the best part of paper
learning: working things out by hand, while adding an intelligent coach that can
generate tasks, inspect reasoning, and respond immediately.

Second, it is an active sports coach. A camera watches the user and a custom
pose-estimation runtime detects body landmarks. The system can use those
landmarks to count squats, push-ups, jumps, and other exercises. This lets the
device count repetitions, notice whether the user is moving, and help make
exercise more regular.

Both parts are controlled through voice, camera, screen, audio, and a single
button, not a keyboard.

## Paper Learning Loop

The system is connected to a plotter: a printer-like mechanism that moves a pen
over paper.

That enables a paper workflow:

1. Gemma 4 creates a task for the student, such as a worksheet.
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

- paper-based learning tasks,
- camera-based grading,
- pen-plotter feedback directly on paper,
- Gemma 4 as the central learning coach and hardware-control brain,
- microphone and camera input with screen, audio, and plotter output,
- local camera-based exercise counting.

Together, these make a system that coaches both intellectual work and physical
activity.
