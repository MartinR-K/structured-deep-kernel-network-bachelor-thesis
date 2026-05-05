Dieses repository enthält alle python-scrips, die in der Bachelorarbeit
"Structured Deep Kernel Networks" von Martin Kolfhaus verwendet wurden.
Für den gridsearch wurden jeweils eine Version mit cross validation für wenige Trainingsepochen,
und eine Version ohne cross validation für Training mit vielen Epochen angelegt.
Die scrips "heat_map.py, heat_map_analysis.py, roc.py, train_history.py"
dienen lediglich zum plotten der Ergebnisse. Das Training wurde auf einer
NVIDIA Grafikkarte durchgeführt.
Es wurden der code von
"https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/sdkn"
benutzt und modifiziert.

Installationsanleitung:

Wegen alter Implementierung des SDKN über pytorch und Kompatibilitätsproblemen mit
dependencies sind alle Versionen in setup.sh gepinnt.Dieses repository enthält alle python-scrips, die in der Bachelorarbeit
"Structured Deep Kernel Networks" von Martin Kolfhaus verwendet wurden.
Für den gridsearch wurden jeweils eine Version mit cross validation für wenige Trainingsepochen,
und eine Version ohne cross validation für Training mit vielen Epochen angelegt.
Die scrips "heat_map.py, heat_map_analysis.py, roc.py, train_history.py"
dienen lediglich zum plotten der Ergebnisse. Das Training wurde auf einer
NVIDIA Grafikkarte durchgeführt.
Es wurden der code von
"https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/sdkn" aus
T. Wenzel and G. Santin and B. Haasdonk, Universality and Optimality of Structured Deep Kernel Networks, ArXiv preprint 2105.07228 (2021)
benutzt und modifiziert.
Diese repository wurde zuletzt am 5. Mai 2026 bearbeitet.

Installationsanleitung:

Wegen alter Implementierung des SDKN über pytorch und Kompatibilitätsproblemen mit
dependencies sind alle Versionen in setup.sh gepinnt.

Installation von conda und venv setup (bash):
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git wget unzip curl

NVIDIA-Treiber überprüfen (bash):
nvidia-smi # sollte GPU anzeigen. Falls nicht, Treiber über den Driver Manager aktualisieren.

Conda installieren (bash):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh

Terminal neu starten oder source ~/.bashrc ausführen, Conda version prüfen (bash):
conda --version

Terminal in Projektverzeichnis (bash):
chmod +x setup.sh
./setup.sh

Bei erfolgreichem setup sollte 
"CUDA available: True
Device: <name>" ausgegeben werden.

Installation von conda und venv setup (bash):
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git wget unzip curl

NVIDIA-Treiber überprüfen (bash):
nvidia-smi # sollte GPU anzeigen. Falls nicht, Treiber über den Driver Manager aktualisieren.

Conda installieren (bash):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh

Terminal neu starten oder source ~/.bashrc ausführen, Conda version prüfen (bash):
conda --version

Terminal in Projektverzeichnis (bash):
chmod +x setup.sh
./setup.sh

Bei erfolgreichem setup sollte 
"CUDA available: True
Device: <name>" ausgegeben werden.

