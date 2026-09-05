## Yleistä

- **Opinto-ohjelma:** TKT

- **Ongelma:** Kuurojen/mykkien on vaikea kommunoida sellaisten kanssa, jotka eivät osaa viittomakieltä. Ratkaisuna tekoälymalli, joka tunnistaa ja muuttaa viitonnan tekstiksi.

- **Ohjelmointikieli:** Python

- **Muut hallitut kielet:** C++, JavaScript

- **Kirjastot:** OpenCV, MediaPipe, NumPy

- **Syöte:** Aluksi ohjelma hyväksyy vain videota/kuvia, myöhemmin pystyy käsittelemään raakaa live-kamerakuvaa. Ohjelma tunnistaa videosta käden/käsien maamerkit, jotka koneoppimismalli luokittelevat kirjaimeksi tai tuntemattomaksi

- **Algoritmit:**

-> Kuvan käsittely OpenCV:llä

-> Käden maamerkkien tunnistus MediaPipella

-> Multi-Layer Perceptron (MLP) staattisia aakkosia varten

-> Backpropagation koulutusalgoritmi Adam optimoijalla

(-> Ajan riittämisestä riippuen Gated Recurrent Unit (GRU) dynaamisia aakkosia varten)

- **Lähteet:**

-> OpenCV video docs: https://docs.opencv.org/4.13.0/dd/d43/tutorial_py_video_display.html

-> MediaPipe hands docs: https://chuoling.github.io/mediapipe/solutions/hands.html

-> NumPy docs: https://numpy.org/doc/stable/

-> Viittomakielen aakkoset datasetti (aakkoset samat ASL kuin suomeksi): https://www.kaggle.com/datasets/datamunge/sign-language-mnist

-> MLP geeksforgeeks (GFG): https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/

-> Adam GFG: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

-> Backpropagation GFG (en käytä lähteenä sivun Python implementaatiota): https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/

(-> GRU GFG: https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/)


## Ydin

Koneoppimismallin rakentaminen ja kouluttaminen backpropagation algoritmilla Adamia käyttäen luokittelemaan viittomakielen aakkosia kuvasta/videosta tunnistetuista käden maamerkeistä.