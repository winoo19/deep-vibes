<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
<h1 align="center">Deep Vibes</h1>
  
https://github.com/user-attachments/assets/88bdeb23-df6a-4234-a65c-aa96f7d3946a

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

We implement different deep learning arquitectures with various degrees of complexity to generate MIDI piano jazz melodies:
- **CNN+VAE**: Variational Autoencoder
- **CNN+LSTM**: Long Short Term Memory
- **CNN+GAN**: Generative Adversarial Network
- **Transformer**: Decoder-only Transformer

Detailed paper can be found in **[/docs/deep-vibes.pdf](./docs/deep-vibes.pdf)**.

<!-- Getting Started -->
## :rocket: Getting Started

To get a local copy up and running follow these simple steps.

1. Clone the repo
   ```sh
   git clone
    ```

2. Install the required packages
    ```sh
    pip install -r requirements.txt
    ```

3. Run the data setup (Optional: Only required for training)
    ```sh
    python -m src.data
    ```

4. Generate melodies
    ```sh
    python -m src.vae.main
    ```

<!-- <p align="right">(<a href="#readme-top">↥ back to top ↥</a>)</p> -->

## 👥 Contributors

<div style="display: flex; justify-content: center;">
  <a href="https://github.com/sergihrs" style="margin: 0px 10px">
    <img src="https://github.com/sergihrs.png" style="border-radius: 50%;" width="50" height="50">
  </a>
  <a href="https://github.com/winoo19" style="margin: 0px 10px">
    <img src="https://github.com/winoo19.png" style="border-radius: 50%;" width="50" height="50">
  </a>
  <a href="https://github.com/nicolasvillagranp" style="margin: 0px 10px">
    <img src="https://github.com/nicolasvillagranp.png" style="border-radius: 50%;" width="50" height="50">
  </a>
  <a href="https://github.com/mariokroll" style="margin: 0px 10px">
    <img src="https://github.com/mariokroll.png" style="border-radius: 50%;" width="50" height="50">
  </a>
</div>
