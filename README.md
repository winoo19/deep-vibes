<a name="readme-top"></a>

<!-- TITLE -->
<h1 align="center">Deep Vibes</h1>

<!-- PROJECT LOGO -->
<div align="center">
  <img src="https://api.junia.ai/storage/v1/object/sign/user-generated-images/18d32214-afe4-4bec-8652-8e6d1b062e19/fce27026-a09f-4199-a48e-80a12810ae8d.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJ1c2VyLWdlbmVyYXRlZC1pbWFnZXMvMThkMzIyMTQtYWZlNC00YmVjLTg2NTItOGU2ZDFiMDYyZTE5L2ZjZTI3MDI2LWEwOWYtNDE5OS1hNDhlLTgwYTEyODEwYWU4ZC5wbmciLCJpYXQiOjE3MDA4ODEyMjAsImV4cCI6MTg1ODU2MTIyMH0.aTQlOzhzCDm_Wnj_tO1kDx-BFr_73tuyOEq6gJdf-gw" alt="Logo" width="350">
</div>

<!-- TABLE OF CONTENTS -->
---

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#contributors">Contributors</a></li>
  </ol>
</details>

---

<br>

<!-- ABOUT THE PROJECT -->
## :memo: About The Project

We implement different deep learning arquitectures to generate
piano melodies.

<p align="right">(<a href="#readme-top">↥ back to top ↥</a>)</p>

<!-- Getting Started -->
## :rocket: Getting Started

To get a local copy up and running follow these simple steps.

### :wrench: Prerequisites

* Python 3.10.6
* Fluidsynth
  - Installation in Windows is done with [Chocolatey](https://chocolatey.org/). 
    ```sh
    choco install fluidsynth
    ```
* Yamaha C5 Grand Piano Soundfont (For generating .wav files)
  - Download the soundfont from [here](https://drive.google.com/file/d/1p0jY3AgGyD9DJGWC25aEUEaydI_n1-3M/view).
  - Extract the soundfont and place it in the `soundfonts` directory.
  ```
  .
  ├── soundfonts
  │   └── Yamaha C5 Grand-v2.4.sf2
  ```

### :hammer: Installation

1. Clone the repo
   ```sh
   git clone
    ```

2. Install the required packages
    ```sh
    pip install -r requirements.txt
    ```

3. Download the Yamaha C5 Grand Piano Soundfont from [here](https://drive.google.com/file/d/1p0jY3AgGyD9DJGWC25aEUEaydI_n1-3M/view) and place it in the `soundfonts` directory.
    ```
    .
    ├── soundfonts
    │   └── Yamaha C5 Grand-v2.4.sf2
    ```

4. Run the following command to generate a melody
    ```sh
    python -m src.generate
    ```

<p align="right">(<a href="#readme-top">↥ back to top ↥</a>)</p>

## 👥 Contributors

<div style="display: flex; justify-content: center;">
  <a href="https://github.com/winoo19" style="margin: 0px 10px">
    <!-- <img src="https://contrib.rocks/image?repo=winoo19/deep-vibes" /> -->
    <img src="https://github.com/winoo19.png" style="border-radius: 50%;" width="50" height="50">
  </a>
  <a href="https://github.com/gomicoder17" style="margin: 0px 10px">
    <!-- <img src="https://contrib.rocks/image?repo=winoo19/deep-vibes" /> -->
    <img src="https://github.com/gomicoder17.png" style="border-radius: 50%;" width="50" height="50">
  </a>
  <a href="https://github.com/nicolasvillagranp" style="margin: 0px 10px">
    <!-- <img src="https://contrib.rocks/image?repo=winoo19/deep-vibes" /> -->
    <img src="https://github.com/nicolasvillagranp.png" style="border-radius: 50%;" width="50" height="50">
  </a>
  <a href="https://github.com/mariokroll" style="margin: 0px 10px">
    <!-- <img src="https://contrib.rocks/image?repo=winoo19/deep-vibes" /> -->
    <img src="https://github.com/mariokroll.png" style="border-radius: 50%;" width="50" height="50">
  </a>
</div>