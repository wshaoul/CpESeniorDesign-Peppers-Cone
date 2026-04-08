# Pepper's Cone Holographic Display

## CEN3907C Computer Engineering Design 1
University of Florida  
Spring 2026  
Team: Steesha D'sa, Gabriel Frank, Sebastian Garcia, Jake Liguori, William Shaoul

This is an open source project for pepper's cone. A developemental "Hologram" for educational use.
We took inspiration from: https://github.com/roxanneluo/Pepper-s-Cone-Unity to get started.

## Project Overview

This repository contains the pre-alpha build of our Pepper's Cone holographic display system. This innovative visualization tool creates 3D hologram-like images by displaying specially processed content on a horizontal screen with a cone-shaped reflector. Our project aims to scale this technology to a 72-inch display for educational environments, enabling remote instructors to appear as life-sized holographic presences in classrooms.

## Completed Work (Pre-Alpha)

### Environment Setup & Research
- Successfully set up development environment with GitHub
- Cloned and explored the past Pepper's Cone repository
- Conducted extensive research on Pepper's Ghost technology and optical illusions
- Reviewed multiple tutorials and academic papers on the technology
- Researched hardware requirements and compatibility issues

### Implementation Progress
- Successfully imported the GitHub repository and got the demo running
- Have begun integrating the Intel RealSense depth camera for a 2.5D hologram
- Started preparing to build a table for a better viewing experience

### Meetings & Planning
- Weekly meetings with our stakeholder and team members to evaluate project updates
- Biweekly meeting with Professor Joe to discuss progress
- Team discussions about Pepper's Cone and how the depth camera will integrate with hardware

## Project Architecture

Our pre-alpha build establishes the foundation for a three-component system:

### 1. Processing System
- Currently using pre-existing demo functionality from the previous team's repository
- Custom processing for video input using OpenCV and MediaPipe

### 2. User Interface
- Ability to upload or livestream input data
- Control features to improve viewing capabilities
- Intuitive UI for easy of use

### 3. Display System
- Successfully tested prohect through demonstration
- Exploring hardware options for modified table, shadowbox, and cone
- Planning pathway to remote connection

## Known Limitations & Challenges

1. **Hardware Integration**: Currently working on the software side to integrate depth camera output to the LIVE application stages.
   - *Next Steps*: Use SDK to feed the output to our application

2. **Image Processing**: Working on converting infrared image output to a 2.5D image for the hologram
   - *Next Steps*: Add infrared image processing code to the repository

3. **Viewing Experience Limitations**: At the moment, demonstrations must be done in a well-lit classroom which ruins the quality of the hologram
   - *Plans*: Build Shadowbox or use tinted cone

4. **Non-funcitonal Buttons**: There are a handful of placeholder buttons
   - *Next Steps*: Program these buttons

## No known Bugs as of right now!

## Setup & Usage

### System Requirements

- GitHub for version control
- Compatible display device for testing

### Installation

1. Install VS Code and Python (version 3.10)
2. Clone the repository:
   ```
   git clone https://github.com/wshaoul/CpESeniorDesign-Peppers-Cone
   ```
3. Run this command in Terminal: 
   ``` 
   cd CpESeniorDesign-Peppers-Cone/Interface
   ```
4. Create Python environment: 
``` 
py -3.10 -m venv winvenv310 
```
5. Run this commmand: 
``` 
.\winvenv310\Scripts\Activate.ps1 
```
6. Install dependencies:
```
pip install opencv-python mediapipe numpy
pip install mediapipe==0.10.21
```
7. Run the LIVE application
```
py studio_main.py
```

## Next Development Priorities

1. Complete table build
2. Develop more robust model/video selection interface
3. Start integrating depth camera ouput
4. Conduct image processing on depth camera output
5. Research and plan for wireless connection

## Team Contributions

The team has collectively invested time in:
- Research and exploration of the Pepper's Cone technology
- Setting up the development environment
- Running and testing the existing demo
- Planning for more hardware integration
- Exploring additional cone and table models

## Contact

For questions about this project, contact the team at:
- sdsa@ufl.edu
- gfrank1@ufl.edu
- s.garcia2@ufl.edu
- jake.liguori@ufl.edu
- wshaoul@ufl.edu

## References

Key resources we've utilized:
1. Original Pepper's Cone GitHub repository: https://github.com/roxanneluo/Pepper-s-Cone-Unity
2. Previous team's Github respository: https://github.com/Rockonmichi/CEN3907C-Pepper-s-Cone

## License

This project is for educational purposes only.
