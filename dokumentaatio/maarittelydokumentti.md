## Yleistä

- **Opinto-ohjelma:** TKT

- **Ongelma:** Kuurojen/mykkien on vaikea kommunoida sellaisten kanssa, jotka eivät osaa viittomakieltä. Ratkaisuna tekoälymalli, joka tunnistaa ja muuttaa viitonnan tekstiksi.

- **Ohjelmointikieli:** Python

- **Muut hallitut kielet:** C++, JavaScript

- **Kirjastot:** OpenCV, MediaPipe, PyTorch

- **Syöte:** Aluksi ohjelma hyväksyy vain videota/kuvia, myöhemmin pystyy käsittelemään raakaa live-kamerakuvaa. Ohjelma tunnistaa videosta käden/käsien maamerkit, jotka koneoppimismalli(t) luokittelevat kirjaimeksi tai tuntemattomaksi

- **Algoritmit:**

-> Kuvan käsittely OpenCV:llä

-> Käden maamerkkien tunnistus MediaPipella

-> Multi-Layer Perceptron (MLP) staattisia aakkosia varten

-> Backpropagation koulutusalgoritmi

-> Ajan riittämisestä riippuen Gated Recurrent Unit (GRU) dynaamisia aakkosia varten

- **Lähteet:**

-> OpenCV video docs: https://docs.opencv.org/4.13.0/dd/d43/tutorial_py_video_display.html

-> MediaPipe hands docs: https://chuoling.github.io/mediapipe/solutions/hands.html

-> Pytorch docs: https://docs.pytorch.org/docs/main/

-> Viittomakielen aakkoset datasetti (aakkoset samat ASL kuin suomeksi): https://www.kaggle.com/datasets/datamunge/sign-language-mnist


## Ydin

Koneoppimismalli(e)n rakentaminen ja kouluttaminen backpropagation algoritmilla luokittelemaan viittomakielen aakkosia kuvasta/videosta tunnistetuista käden maamerkeistä. Datasetin laajentaminen sisältämään dynaamiset aakkoset (J, Z, ääkköset) ehtimisen mukaan